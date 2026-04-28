import pika
import json
import logging
from django.core.management.base import BaseCommand
from core.models import History

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Runs the RabbitMQ consumer to process asynchronous events'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting RabbitMQ consumer..."))
        
        try:
            # Connect to RabbitMQ
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                host='rabbitmq', 
                port=5672,
                # retry connection parameters can be added here if needed
            ))
            channel = connection.channel()
            
            # Ensure the exchange exists
            channel.exchange_declare(exchange='skin_ai_events', exchange_type='topic')
            
            # Create a queue for the history service
            result = channel.queue_declare(queue='history_service_queue', durable=True)
            queue_name = result.method.queue
            
            # Bind the queue to the exchange, listening for 'user.registered' events
            channel.queue_bind(
                exchange='skin_ai_events',
                queue=queue_name,
                routing_key='user.registered'
            )
            
            self.stdout.write(self.style.SUCCESS(f"Waiting for messages on {queue_name}. To exit press CTRL+C"))

            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    event_type = message.get('event')
                    data = message.get('data', {})
                    
                    if event_type == 'user.registered':
                        user_id = data.get('user_id')
                        username = data.get('username')
                        logger.info(f"Received user.registered event for {username} (ID: {user_id})")
                        
                        # Process asynchronous task: Create a welcome history entry
                        welcome_message = {
                            "type": "chat",
                            "custom_title": "Bienvenue sur Skin AI !",
                            "messages": [
                                {
                                    "sender": "bot",
                                    "text": f"Bonjour {username} et bienvenue sur Skin AI ! Votre compte a été créé avec succès. Je suis votre assistant IA, prêt à vous aider avec vos analyses dermatologiques."
                                }
                            ],
                            "top3": [{"label": "Conversation IA", "confidence": 100}]
                        }
                        
                        History.objects.create(
                            user_id=user_id,
                            result=welcome_message
                        )
                        logger.info(f"Successfully created welcome history entry for user {user_id}")
                        self.stdout.write(self.style.SUCCESS(f"Processed welcome entry for user {user_id}"))

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.stdout.write(self.style.ERROR(f"Error processing message: {e}"))
                finally:
                    # Acknowledge the message
                    ch.basic_ack(delivery_tag=method.delivery_tag)

            # Consume messages
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=queue_name, on_message_callback=callback)
            channel.start_consuming()

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"RabbitMQ Connection failed: {e}"))
            logger.error(f"RabbitMQ Connection failed: {e}")

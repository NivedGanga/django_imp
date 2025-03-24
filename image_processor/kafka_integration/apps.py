from django.apps import AppConfig
import threading
from .kafka_utils import start_kafka_consumer

class KafkaIntegrationConfig(AppConfig):
    name = 'kafka_integration'
    def ready(self):
        # Start the Kafka consumer in a separate thread
        kafka_thread = threading.Thread(target=start_kafka_consumer, daemon=True)
        kafka_thread.start()

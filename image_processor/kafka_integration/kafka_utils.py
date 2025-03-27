# kafka_integration/kafka_utils.py
from confluent_kafka import Consumer, KafkaError, KafkaException
from django.conf import settings
import json
import asyncio
from milvus_integration.tasks import getCollection

def start_kafka_consumer():
    conf = {
        'bootstrap.servers': settings.KAFKA_CONFIG["BOOTSTRAP_SERVERS"],
        'group.id': 'image_processing_group',
        'auto.offset.reset': 'earliest'
    }
    print("Starting Kafka consumer...")
    print(settings.KAFKA_CONFIG)
    consumer = Consumer(conf)
    consumer.subscribe([settings.KAFKA_CONFIG["TOPICS"]["image_get"]])
    getCollection()
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    continue
                else:
                    raise KafkaException(msg.error())

            message = msg.value().decode('utf-8')
            asyncio.run(process_message(message))

            # Send acknowledgement
           #send_acknowledgement(msg.value())

    except KeyboardInterrupt:
        print("Consumer interrupted.")

    finally:
        consumer.close()

async def process_message(message):
    from vector_extract.tasks import vectorizeImage
    try:
        message_list = json.loads(message)
        await vectorizeImage(message_list)
        print(f"message processed")
        send_acknowledgement('message')
    except json.JSONDecodeError as e:
        print(f"Failed to decode message: {e}")

def send_acknowledgement(message):
    # Add logic to send acknowledgement
    from confluent_kafka import Producer
    producer = Producer({
        'bootstrap.servers': settings.KAFKA_CONFIG["BOOTSTRAP_SERVERS"]
    })

    ack_topic = settings.KAFKA_CONFIG["TOPICS"]["ack"]
    producer.produce(ack_topic, value=f"Processed: {message}")
    producer.flush()
    print(f"Sent acknowledgement for: {message}")
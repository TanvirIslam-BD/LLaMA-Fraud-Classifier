from celery import shared_task

from ml_app.ml.train import train_model_cron


@shared_task
def auto_learning_task():
    try:
        # pull JSON data with API call (replace with your JSON source)
        json_data = [
            {
                "Customer": "John Doe",
                "Organisation": "EventCorp",
                "Event": "MusicFest2024",
                "Amount": 120.5,
                "Tickets": 2,
                "Service Charge": 5.0,
                "Coupon amount": 10.0,
                "Booking type": "Online",
                "Status": "Confirmed",
                "Processor": "PayPal",
                "Currency": "USD",
                "Date": "2024-11-22T14:30:00",
                "Genuine Order": 1
            },
            {
                "Customer": "Jane Smith",
                "Organisation": "FunEvents",
                "Event": "FoodFest2024",
                "Amount": 80.0,
                "Tickets": 1,
                "Service Charge": 3.0,
                "Coupon amount": 5.0,
                "Booking type": "Offline",
                "Status": "Pending",
                "Processor": "Stripe",
                "Currency": "USD",
                "Date": "2024-11-22T16:45:00",
                "Genuine Order": 0
            }
        ]

        # Process each JSON object for incremental learning
        train_model_cron(json_data)
    except Exception as e:
        print(f"Error in IncrementalLearningCronJob: {str(e)}")

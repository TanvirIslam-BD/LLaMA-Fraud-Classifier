from celery import shared_task

from ml_app.ml.train import train_model_cron


@shared_task
def auto_learning_task():
    try:
        # Pull JSON data with API call (replace with your JSON source)
        print("Model Learning......")

        json_data = [
            {
                "Customer": "Chinedu  J. Orjiudeh",
                "Organisation": "OCTA HQ -lifeoftwinklee",
                "Event": "LUNGU RAVE",
                "Processor": "Cash",
                "Booking type": "Cash",
                "Refund status": "No Refund",
                "Currency": "AUD",
                "Status": "Active",
                "Date": "11/3/2024  12:08:00 AM",
                "Tickets": 1,
                "Amount": 0,
                "Service Charge": 0,
                "Coupon amount": 0,
                "Genuine Order": 1
            },
            {
                "Customer": "Edward Beaurepaire",
                "Organisation": "John Newman Lawes",
                "Event": "Urban and Retreat",
                "Processor": "SecurePay JS SDK",
                "Booking type": "Card",
                "Refund status": "Refunded",
                "Currency": "AUD",
                "Status": "Cancelled",
                "Date": "6/2/2024  6:54:00 AM",
                "Tickets": 3,
                "Amount": 675.00,
                "Service Charge": 14.40,
                "Coupon amount": 0,
                "Genuine Order": 0
            }
        ]

        # Process each JSON object for incremental learning
        # Commented out for now to prevent model overfitting and instability.
        train_model_cron(json_data)
    except Exception as e:
        print(f"Error in IncrementalLearningCronJob: {str(e)}")

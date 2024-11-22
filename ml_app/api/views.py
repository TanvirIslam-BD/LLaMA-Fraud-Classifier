import json
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from ml_app.ml.predict import prediction

class PredictAPIView(APIView):
    permission_classes = [IsAuthenticated]  # Require authentication, can be modified as needed

    def post(self, request):
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body)

            # Convert the data to a DataFrame with a single row
            data_df = pd.DataFrame([data])

            result = prediction(data_df)  # Calls the function from `predict.py`

            # Return the prediction result
            return Response({"prediction": result.tolist()}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

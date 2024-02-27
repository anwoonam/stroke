from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .models import UserData1, UserData2
from .predictor import MLModel

# Create your views here.

class OcPredictAPIView(APIView):

#     def get(self, request):
#         return Response({
#             'intervention_1_options': self.intervention_1_options,
#             'intervention_2_options': self.intervention_2_options,
#             'cate_dic': self.cate_dic,
#         }, status=200 )

    def post(self, request):
        age = request.data.get('age')
        gender = request.data.get('gender')
        bmi = request.data.get('bmi')
        session_n = request.data.get('session_n')
        training_p = request.data.get('training_p')
        training_t = request.data.get('training_t')
        intervention_1 = request.data.get('intervention_1')
        intervention_2 = request.data.get('intervention_2')

        if not all([age, gender, bmi, session_n, training_p, training_t, intervention_1, intervention_2]):
            return Response({'error': 'Missing required fields'}, status=status.HTTP_400_BAD_REQUEST)

        # 저장된 데이터베이스에 사용자 데이터 저장
        user_data = UserData1(age=age, gender=gender, bmi=bmi, session_n=session_n,
                              training_p=training_p, training_t=training_t,
                              intervention_1=intervention_1, intervention_2=intervention_2)
        user_data.save()

        # 머신러닝 모델을 통해 예측
        prediction = MLModel.oc_predict(age,
                                        bmi,
                                        intervention_2,
                                        session_n,
                                        training_p,
                                        training_t,
                                        gender
                                        )

        return Response({
            'prediction': prediction
        }, status=200 )

class IvPredictAPIView(APIView):

    def post(self, request):
        # POST 요청에 대한 처리
        age = request.data.get('age')
        gender = request.data.get('gender')
        bmi = request.data.get('bmi')
        session_n = request.data.get('session_n')
        training_p = request.data.get('training_p')
        training_t = request.data.get('training_t')
        outcome_1 = request.data.get('outcome_1')
        outcome_2 = request.data.get('outcome_2')

        if not all([age, gender, bmi, session_n, training_p, training_t, outcome_1, outcome_2]):
            return Response({'error': 'All fields are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 저장된 데이터베이스에 사용자 데이터 저장
        user_data = UserData2(age=age, gender=gender, bmi=bmi, session_n=session_n,
                              training_p=training_p, training_t=training_t,
                              outcome_1=outcome_1, outcome_2=outcome_2)
        user_data.save()

        # 머신러닝 모델을 통해 예측
        prediction = MLModel.iv_predict(age,
                                        bmi,
                                        outcome_2,
                                        session_n,
                                        training_p,
                                        training_t,
                                        gender
                                        )

        return Response({
            'prediction': prediction
        }, status=200 )
    
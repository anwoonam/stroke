from django.urls import path
from .views import IvPredictAPIView, OcPredictAPIView

urlpatterns = [
    path('iv_oc/', OcPredictAPIView.as_view()),
    path('oc_iv/', IvPredictAPIView.as_view()),
]
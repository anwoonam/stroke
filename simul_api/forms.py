from django import forms
from .models import UserData1, UserData2

# Create your forms here.

class UserDataForm1(forms.ModelForm):
    class Meta:
        model = UserData1
        fields = ['age', 'gender', 'bmi', 'session_n', 'training_p', 'training_t',
                  'intervention_1', 'intervention_2']

class UserDataForm2(forms.ModelForm):
    class Meta:
        model = UserData2
        fields = ['age', 'gender', 'bmi', 'session_n', 'training_p', 'training_t',
                  'outcome_1', 'outcome_2']
        
from django.db import models

# Create your models here.
class UserData1(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=4)
    bmi = models.FloatField()
    session_n = models.IntegerField()
    training_p = models.IntegerField()
    training_t = models.IntegerField()
    intervention_1 = models.CharField(max_length=200)
    intervention_2 = models.CharField(max_length=200)

    def __str__(self):
        return f'{self.age} | {self.gender} | {self.bmi} | {self.session_n} | {self.training_p} | {self.training_t} | {self.intervention_1} | {self.intervention_2}'
    
class UserData2(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=4)
    bmi = models.FloatField()
    session_n = models.IntegerField()
    training_p = models.IntegerField()
    training_t = models.IntegerField()
    outcome_1 = models.CharField(max_length=200)
    outcome_2 = models.CharField(max_length=200)

    def __str__(self):
        return f'{self.age} | {self.gender} | {self.bmi} | {self.session_n} | {self.training_p} | {self.training_t} | {self.outcome_1} | {self.outcome_2}'
    
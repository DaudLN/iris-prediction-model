from django.urls import path
from . import views

# app_name = "predict"
urlpatterns = [
    path("", views.predict, name="predict"),
    path("result/", views.result, name="result"),
    path("results/", views.view_results, name="results"),
]

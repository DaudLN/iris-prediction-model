from django.shortcuts import render
from django.http import JsonResponse
import joblib
from .models import PredictionResult

# Create your views here.
def predict(request):
    return render(request, "predict.html")


def result(request):
    if request.method == "POST":
        # Receive data from client
        sepal_length = float(request.POST.get("sepal_length"))
        sepal_width = float(request.POST.get("sepal_width"))
        petal_length = float(request.POST.get("petal_length"))
        petal_width = float(request.POST.get("petal_width"))
        results = dict(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        model = joblib.load("./model/model.joblib")
        prediction = model.predict(
            [[sepal_length, sepal_width, petal_length, petal_width]]
        )
        PredictionResult.objects.create(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            classification=prediction[0],
        )
        print(prediction)
        context = dict(results=results, prediction=prediction[0])
        return JsonResponse(context)
    else:
        return render(request, "result.html")


def view_results(request):
    dataset = PredictionResult.objects.all()
    context = dict(dataset=dataset)
    return render(request, "result.html", context)

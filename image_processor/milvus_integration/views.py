from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from milvus_integration.tasks import get_all_data,delete_collection

def get_datas(request):
    results = get_all_data(request)
    return render(request, 'results_table.html', {'results': results})

@csrf_exempt 
@require_http_methods(["DELETE"])   
def delete_col(request):
    delete_collection()
    return JsonResponse({'message': 'Collection deleted.'})
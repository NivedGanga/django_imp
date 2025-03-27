"""
URL configuration for image_processor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from milvus_integration import views as milvus
from cluster import views as cluster
urlpatterns = [
    path('admin/', admin.site.urls),
    path('milvusdatas/', milvus.get_datas),
    path('deletecollection', milvus.delete_col),
    path('upload',cluster.process_image),
]

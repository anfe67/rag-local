from django.urls import path

from . import views

urlpatterns = [
    path("", views.documents_page, name="home"),
    path("documents/", views.documents_page, name="documents-page"),
    path("chat/", views.chat_page, name="chat-page"),
    path("api/documents/", views.document_api, name="document-api"),
    path("api/chat/", views.chat_api, name="chat-api"),
]

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import UploadImageModel
from .serializers import UploadImageSerializer
from rest_framework.parsers import MultiPartParser, FormParser

class ImgUploadAPIView(APIView):
    authentication_classes = []
    permission_classes = []
    parser_classes = [MultiPartParser, FormParser]  # handle form-data + files

    def post(self, request):
        # Just pass request.data; it contains both caption and image
        serializer = UploadImageSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Media Uploaded Successfully", "data": serializer.data},
                status=status.HTTP_201_CREATED
            )
        return Response(
            {"message": serializer.errors, "data": None},
            status=status.HTTP_400_BAD_REQUEST
        )

    def get(self, request):
        qs = UploadImageModel.objects.all().order_by("-id")
        serializer = UploadImageSerializer(qs, many=True, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)
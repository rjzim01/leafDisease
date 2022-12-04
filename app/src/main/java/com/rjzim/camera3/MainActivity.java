package com.rjzim.camera3;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.rjzim.camera3.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    ImageView imgView;
    Button btnCamera, btnGallery, btnDetect;
    TextView text,km,r;

    int imageSize = 224; //default image size

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Objects.requireNonNull(getSupportActionBar()).hide();

        imgView = findViewById(R.id.img_view);
        btnCamera = findViewById(R.id.btn_camera);
        btnGallery = findViewById(R.id.btn_gallery);
        btnDetect = findViewById(R.id.btn_detect);
        text = findViewById(R.id.text);
        km = findViewById(R.id.km);
        r= findViewById(R.id.r);

        km.setVisibility(View.GONE);
        r.setVisibility(View.GONE);

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent, 1);
                    }else {
                        //request camera permission if we don't have
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                    }
                }
            }
        });

        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Open Gallery

                Intent i = new Intent();
                i.setAction(Intent.ACTION_GET_CONTENT);
                i.setType("image/*");

                startActivityForResult(i,10);
            }
        });

        btnDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                text.setVisibility(View.VISIBLE);
                km.setVisibility(View.VISIBLE);
                r.setVisibility(View.VISIBLE);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (requestCode == 1 && resultCode == RESULT_OK){
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imgView.setImageBitmap(image);
            text.setVisibility(View.GONE);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }

        else if (requestCode == 10) {
            if (data!=null){
                Uri uri = data.getData();
                try {
                    Bitmap image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    int dimension = Math.min(image.getWidth(), image.getHeight());
                    image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                    imgView.setImageBitmap(image);
                    text.setVisibility(View.GONE);

                    image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                    classifyImage(image);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
        try {
            // model line 1
            //ModelUnquant model = ModelUnquant.newInstance(context);
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // model line 2
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            //
            //====convert a bitmap into ByteBuffer=====
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            //
            //resize the image because our model expects the exact input shape (224x224 pixels),
            // get id array of 224 * 224 pixels in image
            int[] intValue = new int[imageSize * imageSize];
            image.getPixels(intValue, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, B values , add to byteBuffer
            int pixel = 0;
            for (int i = 0; i < imageSize; i++){
                for (int j = 0; j < imageSize; j++){
                    int val = intValue[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }
            //==============================================

            //model line 3
            inputFeature0.loadBuffer(byteBuffer);

            //model line 4
            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //
            float[] confidence = outputFeature0.getFloatArray();

            //find the index of the class with the biggest confidence
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i =0;i<confidence.length; i++){
                if (confidence[i] > maxConfidence){
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Pepper bell bacterial spot", "Pepper bell healthy", "Potato early blight", "Potato healthy", "Potato late blight", "Tomato target spot", "Tomato mosaic virus", "Tomato yellow leaf curl virus", "Tomato bacterial spot", "Tomato early blight", "Tomato healthy", "Tomato late blight", "Tomato leaf mold", "Tomato septoria leaf spot", "Tomato spider mites two spotted spider mite"};
            text.setVisibility(View.GONE);
            km.setVisibility(View.GONE);
            r.setVisibility(View.GONE);

            text.setText(classes[maxPos]);
            text.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    // to search the disease on internet
                    startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=" + text.getText())));
                }
            });

            //model line 5
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
















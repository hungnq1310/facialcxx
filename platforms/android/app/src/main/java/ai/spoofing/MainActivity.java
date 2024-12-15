// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.spoofing;
import ai.spoofing.databinding.ActivityMainBinding;
import ai.spoofing.*;

import android.content.DialogInterface;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.os.Bundle;
import android.util.Log;

import android.view.View;
import android.widget.Button;
import android.widget.ImageView;


import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import kotlinx.coroutines.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;


class Result {
    public List<Integer> detectedIndices;
    public List<Float> detectedScore;
    public long processTimeMs;

    public Result(List<Integer> detectedIndices, List<Float> detectedScore, long processTimeMs) {
        this.detectedIndices = detectedIndices;
        this.detectedScore = detectedScore;
        this.processTimeMs = processTimeMs;
    }
}

public class MainActivity extends AppCompatActivity {
    ActivityMainBinding binding;

    private ExecutorService backgroundExecutor = Executors.newSingleThreadExecutor();
    private List<String> labelData;

    private static final String TAG = "ORTSpoofing";

    private static final double SCORE_SPOOF = 0.6345;

    static {
        System.loadLibrary("spoofing");
        System.loadLibrary("ortcxx");
        System.loadLibrary("onnxruntime");
    };

    private Button uploadButton;
    private View myview;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot()); // Adjust layout reference if needed

        // Init model
        try {
            labelData = readLabels();
            String yoloPath = "weights/yolov7-tiny-v0.onnx";
            String extractorPath = "weights/extractor.onnx";
            String embedderPath = "weights/embedder.onnx";
            TensorUtils.initModel(getAssets(), yoloPath, extractorPath, embedderPath);
        } catch (Exception e) {
            Log.e(TAG, "Error initializing models", e);
        }

        uploadButton = findViewById(R.id.button2);
        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Show a dialog or list to choose from assets
                showAssetPicker();
            }
        });
    }

    private void showAssetPicker() {
        // Get a list of asset filenames
        String[] assetFiles = null;
        try {
            assetFiles = getAssets().list("tests/"); // List all files in assets folder
            for (String file : assetFiles) {
                Log.d("Assets", file);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error listing assets", e);
            return;
        }

        // Filter for images and videos
        List<String> imageVideoFiles = new ArrayList<>();
        for (String file : assetFiles) {
            if (file.endsWith(".jpg") || file.endsWith(".png") || file.endsWith(".mp4")) {
                imageVideoFiles.add(file);
            }
        }

        // Create and show the dialog
        new AlertDialog.Builder(this)
                .setTitle("Choose an Asset")
                .setItems(imageVideoFiles.toArray(new String[0]), new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        String selectedAsset = imageVideoFiles.get(which);
                        // Handle the selected asset
                        handleSelectedAsset(selectedAsset);
                    }
                })
                .show();
    }

    private void handleSelectedAsset(String assetFilename) {
        try {
            if (assetFilename.endsWith(".jpg") || assetFilename.endsWith(".png")) {
                //set video view to gone
                findViewById(R.id.MainVideoView).setVisibility(View.GONE);
                findViewById(R.id.MainImageView).setVisibility(View.VISIBLE);

                // Load image into ImageView
                InputStream is = getAssets().open("tests/" + assetFilename);
                //read byte from input stream
                byte[] bytes = new byte[is.available()];
                is.read(bytes);
                is.close();

                //process image and display
                byte[] processed_bytes = processed_bytes(bytes, true);

                //inference
                processImage(processed_bytes);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error loading asset", e);
            return outputFile.getAbsolutePath();
        }
    }

    private void processImage(byte[] image) {
        checkClearUI();
        new Thread(() -> {
            // ... (Process inference results and update UI)
            if (image != null) {
                // Create a Result object and update the UI
                float percentage =  100;
                InferenceBytes(image, percentage);
            }
        }).start();
    }

        
    private void checkClearUI(){
        binding.detectedItemValue3.setVisibility(View.GONE);
        binding.detectedItemValue2.setVisibility(View.GONE);
        binding.detectedItemValue1.setVisibility(View.GONE);
        binding.imageDetection1.setVisibility(View.GONE);
        binding.imageDetection2.setVisibility(View.GONE);
        binding.imageDetection3.setVisibility(View.GONE);
    }


    private void updateUI(DetectionResult[] results, long processTimeMs, float percentage) {
        if (results.length == 0) {
            return ;
        }

        runOnUiThread(() -> {
            Log.d("Percentage", String.valueOf((int) percentage));
            binding.percentMeter.setProgress((int) percentage, true);

            float score = results[0].probs[0];
            String label = score < SCORE_SPOOF ? labelData.get(0) : labelData.get(1);
            binding.detectedItem1.setText(label);
            binding.detectedItemValue1.setText(String.format("%.4f%%", score * 100f));
            binding.detectedItemValue1.setVisibility(View.VISIBLE);
            binding.imageDetection1.setImageBitmap(results[0].faceBitmap);
            binding.imageDetection1.setVisibility(View.VISIBLE);

            if (results.length > 1) {
                float score2 = results[1].probs[0];
                String label2 = score2 < SCORE_SPOOF ? labelData.get(0) : labelData.get(1);
                binding.detectedItem2.setText(label2);
                binding.detectedItemValue2.setText(String.format("%.4f%%", score2 * 100f));
                binding.detectedItemValue2.setVisibility(View.VISIBLE);
                binding.imageDetection2.setImageBitmap(results[1].faceBitmap);
                binding.imageDetection2.setVisibility(View.VISIBLE);
            }

            if (results.length > 2) {
                float score3 = results[2].probs[0];
                String label3 = score3 < SCORE_SPOOF ? labelData.get(0) : labelData.get(1);
                binding.detectedItem3.setText(label3);
                binding.detectedItemValue3.setText(String.format("%.4f%%", score3 * 100f));
                binding.detectedItemValue3.setVisibility(View.VISIBLE);
                binding.imageDetection3.setImageBitmap(results[2].faceBitmap);
                binding.imageDetection3.setVisibility(View.VISIBLE);
            }

            binding.inferenceTimeValue.setText(String.format("%dms", processTimeMs));
        });
    }

    private List<String> readLabels() {
        List<String> labels = new ArrayList<>();
        try (InputStream inputStream = getResources().openRawResource(R.raw.spoofing_label);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading labels", e);
        }
        return labels;
    };

    private byte[] processed_bytes(byte[] data, boolean display_image) {
        try {
            // Get the rotation from Exif metadata
            ExifInterface exifInterface = new ExifInterface(new ByteArrayInputStream(data));
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

            // Determine the rotation in degrees
            int rotationDegrees = 0;
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotationDegrees = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotationDegrees = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotationDegrees = 270;
                    break;
            }

            // Convert byte[] to Bitmap
            Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length);

            // Rotate the Bitmap if needed
            Bitmap rotatedBitmap = rotateBitmap(bitmap, rotationDegrees);

            // Use the rotatedBitmap (e.g., display it or save it)
            // Example: Display in an ImageView
            if (display_image){
                displayImage(rotatedBitmap);
            }

            // Convert the rotated image as byte array
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream); // reserve full quality
            byte[] rotatedData = outputStream.toByteArray();

            // Send rotatedData for model inference
            return rotatedData;
        } catch (IOException e) {
            Log.e(TAG, "Error processing image", e);
        }
        return data;
    }

    
    // Helper function to rotate a Bitmap
    private Bitmap rotateBitmap(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private void displayImage(Bitmap bitmap) {
        findViewById(R.id.MainImageView).setImageBitmap(bitmap);
    }

    // update UI
    private void Inference(byte[] byteArray, float percentage) {

        long time = System.currentTimeMillis();
        DetectionResult[] detectionResults = TensorUtils.checkspoof(byteArray);
        long processTimeMs = System.currentTimeMillis() - time;

        if (detectionResults != null) {
            updateUI(detectionResults, processTimeMs, percentage);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
    }
}
package ai.spoofing;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

public class TensorUtils {
    // inference in C++
    public static native void initModel(
        AssetManager assetManager,
        String yoloPath,
        String extractorPath,
        String embedderPath
    );
    public static  native DetectionResult[] checkspoof(byte[] imageArray);
}


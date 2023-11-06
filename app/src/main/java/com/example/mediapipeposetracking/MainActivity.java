package com.example.mediapipeposetracking;

import static java.lang.Float.max;
import static java.lang.Float.min;

import android.annotation.SuppressLint;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.SurfaceTexture;
import android.os.Build;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;

import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.glutil.EglManager;
import com.google.protobuf.InvalidProtocolBufferException;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONException;
import android.content.res.AssetManager;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Main activity of MediaPipe example apps.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Test";
    private static final String BINARY_GRAPH_NAME = "pose_tracking_gpu.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME = "pose_landmarks";

    // private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.FRONT;
    private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.BACK;
    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;
    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private FrameProcessor processor;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;
    // ApplicationInfo for retrieving metadata defined in the manifest.
    private ApplicationInfo applicationInfo;
    // Handles camera access via the {@link CameraX} Jetpack support library.
    private CameraXPreviewHelper cameraHelper;
    private TextView title;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(getContentViewLayoutResId());

        title = findViewById(R.id.title1);
        title.bringToFront();

        // Static Params.
        float Threshold = 0.2f;
        int N = 5;
        String actionFile = "KHT.json";

        try {
            applicationInfo =
                    getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
        } catch (NameNotFoundException e) {
            Log.e(TAG, "Cannot find application info: " + e);
        }

        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();//设置相机预览到ui上

        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        BINARY_GRAPH_NAME,
                        INPUT_VIDEO_STREAM_NAME,
                        OUTPUT_VIDEO_STREAM_NAME);
        processor
                .getVideoSurfaceOutput()
                .setFlipY(FLIP_FRAMES_VERTICALLY);

        PermissionHelper.checkAndRequestCameraPermissions(this);
        AndroidPacketCreator packetCreator = processor.getPacketCreator();
        Map<String, Packet> inputSidePackets = new HashMap<>();
//        inputSidePackets.put(INPUT_NUM_HANDS_SIDE_PACKET_NAME, packetCreator.createInt32(NUM_HANDS));
//        processor.setInputSidePackets(inputSidePackets);


        // Load KNNArgs.
        StringBuilder stringBuilder = new StringBuilder();
        try {
            AssetManager assetManager = this.getAssets();
            BufferedReader bf = new BufferedReader(new InputStreamReader(
                    assetManager.open("KNNArgs.json")
            ));
            String line;
            while ((line = bf.readLine()) != null) {
                stringBuilder.append(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "IO Fail: " + e);
            return;
        }
        String jsonString = stringBuilder.toString();

        JSONArray jarr = null;
        JSONArray jarr1 = null;
        try {
            jarr = new JSONArray(jsonString);
            jarr1 = jarr.getJSONArray(0);
        } catch (JSONException e) {
            Log.e(TAG, "Json Fail: " + e);
            return;
        }
        // Log.i(TAG, "jarr len: " + jarr.length());
        // Log.i(TAG, "jarr0 len: " + jarr1.length());

        // dumplcate test.
        // float[][] KNNArgs = new float[jarr.length() * 2][jarr1.length()];
        float[][] KNNArgs = new float[jarr.length()][jarr1.length()];
        for (int i = 0; i < jarr.length(); i++) {
            try {
                JSONArray innerJsonArray = jarr.getJSONArray(i);
                for (int j = 0; j < innerJsonArray.length(); j++) {
                    KNNArgs[i][j] = (float) innerJsonArray.getDouble(j);
                    // dumplcate test.
                    // KNNArgs[i*2][j] = (float) innerJsonArray.getDouble(j);
                    // KNNArgs[i*2+1][j] = (float) innerJsonArray.getDouble(j);
                }
            } catch (JSONException e) {
                Log.e(TAG, "onCreate: Json Detach error", e);
            }
        }
        // Load KNNArgs over.


        // Load Action.
        JSONArray actionArr = null;
        try {
            StringBuilder ActionBuilder = new StringBuilder();
            AssetManager assetManager = this.getAssets();
            BufferedReader bf = new BufferedReader(new InputStreamReader(
                    assetManager.open(actionFile)
            ));
            String line;
            while ((line = bf.readLine()) != null) {
                ActionBuilder.append(line);
            }
            actionArr = new JSONArray(ActionBuilder.toString());
        } catch (IOException e) {
            Log.e(TAG, "IO Fail: " + e);
            return;
        } catch (JSONException e) {
            Log.e(TAG, "Action Load Fail: " + e);
            return;
        }

        int[] action_list = new int[actionArr.length()];
        for (int i = 0; i < actionArr.length(); i++) {
            try {
                action_list[i] = actionArr.getInt(i);
            } catch (JSONException e) {
                Log.e(TAG, "onCreate: Json Detach error", e);
            }
        }


        // To show verbose logging, run:
        // adb shell setprop log.tag.MainActivity VERBOSE
//        if (Log.isLoggable(TAG, Log.VERBOSE)) {

        // State Params.
        AtomicInteger state_index = new AtomicInteger();
        AtomicInteger action_count = new AtomicInteger();
        // Log.i("RESULT", "Action Count: " + action_count.get());
        Log.i("CHANGE_RESULT", "Action Count: " + action_count.get());
        title.bringToFront();
        title.setText("Count:" + action_count.get());

        processor.addPacketCallback(
                OUTPUT_LANDMARKS_STREAM_NAME,
                (packet) -> {
                    Log.v(TAG, "Received multi-hand landmarks packet.");

                    Log.v(TAG, packet.toString());
                    byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                    try {
                        NormalizedLandmarkList landmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                        if (landmarks == null) {
                            // Log.v(TAG, "[TS:" + packet.getTimestamp() + "] No iris landmarks.");
                            return;
                        }
                        // Note: If eye_presence is false, these landmarks are useless.
//                    Log.v(
//                            TAG,
//                            "[TS:"
//                                    + packet.getTimestamp()
//                                    + "] #Landmarks for iris: "
//                                    + landmarks.getLandmarkCount());
//                    Log.v(TAG, getLandmarksDebugString(landmarks));
                        // Log.i("TS", "[TS:" + packet.getTimestamp() + "]");


                        // Get Pose Data of this man.
                        float[] feature = new float[24];
                        //肩膀、胯、膝盖、脚踝、手肘、手腕
                        int[] select_indexs = {11, 12, 23, 24, 25, 26, 27, 28, 13, 14, 15, 16};
                        for (int i = 0; i < 24; i++) {
                            feature[i] = 0.0f;
                        }
                        if (landmarks.getLandmarkCount() > 0) {
                            // landmarks.getLandmark(0).getX();
                            for (int i = 0; i < 12; i++) {//以此拿到各部位的坐标数据，偶数下标是x,奇数下标是y
                                feature[i * 2] = landmarks.getLandmark(select_indexs[i]).getX();
                                feature[i * 2 + 1] = landmarks.getLandmark(select_indexs[i]).getY();
                            }

                            float minx = feature[0], maxx = feature[0];
                            float miny = feature[1], maxy = feature[1];
                            for (int i = 2; i < 24; i += 2) {//获取到数据包最大和最小的x y坐标
                                maxx = max(maxx, feature[i]);
                                minx = min(minx, feature[i]);
                                maxy = max(maxy, feature[i + 1]);
                                miny = min(miny, feature[i + 1]);
                            }

                            for (int i = 0; i < 12; i++) {//这个用来干嘛呢？？？
                                feature[i * 2] = (feature[i * 2] - minx) / (maxx - minx);
                                feature[i * 2 + 1] = (feature[i * 2 + 1] - miny) / (maxy - miny);
                            }
                        }


                        // Calculate the Nearest Pose.
                        ArrayList<Pair<Double, Integer>> distances = new ArrayList<>(); //距离
                        for (float[] knnArg : KNNArgs) {
                            double distance = 0;
                            for (int j = 0; j < feature.length; j++) {
                                distance += Math.pow(feature[j] - knnArg[j], 2);//平方累加 干嘛用？？？
                            }
                            if (distance <= Threshold) {//算出来的值如果小于阀值
                                distances.add(new Pair<>(distance, (int) knnArg[knnArg.length - 1]));
                            }
                        }
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                            //排序
                            distances.sort(Comparator.comparingDouble(o -> o.first));
                        } else {
                            Log.e(TAG, "onCreate: SDK Version too lower.");
                            return;
                        }

                        HashMap<Integer, Integer> labels_count = new HashMap<>();
                        Pair<Integer, Integer> max_label = new Pair<>(-1, 0);
                        for (int i = 0; i < Math.min(N, distances.size()); i++) {//循环距离集合，最少循环N次
                            int label = distances.get(i).second;
                            labels_count.put(label, labels_count.getOrDefault(label, 0) + 1);//从这里开始看不懂！！！！！
                            if (labels_count.get(label) > max_label.second) {
                                max_label = new Pair<>(label, labels_count.get(label));
                            }
                        }


                        // Match this pose.
                        Log.i("POSE", "Detect Post Index: " + max_label.first);
                        int label = max_label.first;
                        final int th_state = state_index.get();
                        if (label == action_list[th_state]) {
                            state_index.addAndGet(1);
                        } else if (label == -1 || (th_state != 0 && label == action_list[th_state - 1])) {
                            // do nothing.
                        } else {
                            state_index.set(0);
                        }

                        if (state_index.get() == action_list.length) {//如果视频流获取到的动作和模型符合，数量+1
                            action_count.addAndGet(1);
                            state_index.set(0);
                            // Log.i("RESULT", "Action Count: " + action_count.get());
                            Log.i("CHANGE_RESULT", "Action Count: " + action_count.get());
                            // title.bringToFront();
                            title.setText("Count:" + action_count.get());
                        }
                    } catch (InvalidProtocolBufferException e) {
                        Log.e(TAG, "Couldn't Exception received - " + e);
                        return;
                    }
                }
        );
//        }
    }

    private static String getLandmarksDebugString(NormalizedLandmarkList landmarks) {
        int landmarkIndex = 0;
        String landmarksString = "";
        for (LandmarkProto.NormalizedLandmark landmark : landmarks.getLandmarkList()) {
            landmarksString +=
                    "\t\tLandmark["
                            + landmarkIndex
                            + "]: ("
                            + landmark.getX()
                            + ", "
                            + landmark.getY()
                            + ", "
                            + landmark.getZ()
                            + ")\n";
            ++landmarkIndex;
        }
        return landmarksString;
    }

    // Used to obtain the content view for this application. If you are extending this class, and
    // have a custom layout, override this method and return the custom layout.
    protected int getContentViewLayoutResId() {
        return R.layout.activity_main;
    }

    @Override
    protected void onResume() {
        super.onResume();
        converter =
                new ExternalTextureConverter(
                        eglManager.getContext(), 2);
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();

        // Hide preview display until we re-open the camera again.
        previewDisplayView.setVisibility(View.GONE);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        previewFrameTexture = surfaceTexture;
        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    protected Size cameraTargetResolution() {
        return null; // No preference and let the camera (helper) decide.
    }

    public void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    onCameraStarted(surfaceTexture);
                });
        CameraHelper.CameraFacing cameraFacing = CAMERA_FACING;
        cameraHelper.startCamera(
                this, cameraFacing, /*unusedSurfaceTexture=*/ null, cameraTargetResolution());
    }

    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(
            SurfaceHolder holder, int format, int width, int height) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        boolean isCameraRotated = cameraHelper.isCameraRotated();

        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }

    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }
}
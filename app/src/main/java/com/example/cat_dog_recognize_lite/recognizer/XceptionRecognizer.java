package com.example.cat_dog_recognize_lite.recognizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.example.cat_dog_recognize_lite.utils.ImageProcess;
import com.example.cat_dog_recognize_lite.utils.Recognition;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.PrimitiveIterator;

public class XceptionRecognizer {

    public static class LabelInfo {
        public int labelId;
        public String labelName;
        public float labelScore;

        public LabelInfo(int labelId, String labelName, float labelScore) {
            this.labelId = labelId;
            this.labelName = labelName;
            this.labelScore = labelScore;
        }
    }

    private final Size INPNUT_SIZE = new Size(320, 320);
    private final int[] OUTPUT_SIZE = new int[]{1, 200};
    private Boolean IS_INT8 = false;
    private final float THRESHOLD = 0.25f;
    ImageProcess imageProcess = new ImageProcess();

    private final String LABEL_FILE = "cat_dog_breeds_cn.txt";
    private String FP16_MODEL_FILE = "cat_dog_classify_xception_0407_fp16.tflite";
    private String INT8_MODEL_FILE = "cat_dog_classify_xception_0407_int8.tflite";
    private String MODEL_FILE;

    // ?????????????????????int8 quant??????, ????????????input/output Tensor?????????
//    private int QUANT_MODEL_INPUT_TENSOR_INDEX = 231;
//    private int QUANT_MODEL_OUTPUT_TENSOR_INDEX = 232;
    MetadataExtractor.QuantizationParams inputQuantParams = new MetadataExtractor.QuantizationParams(0.00392157f, 0);
    MetadataExtractor.QuantizationParams outputQuantParams = new MetadataExtractor.QuantizationParams(0.00390625f, 0);
    private Interpreter tflite;
//    private MetadataExtractor tfliteMeta;
    private List<String> associatedAxisLabels;
    Interpreter.Options options = new Interpreter.Options();


    public String getLabelFile() {
        return this.LABEL_FILE;
    }
    public Size getInputSize(){return this.INPNUT_SIZE;}
    public int[] getOutputSize(){return this.OUTPUT_SIZE;}
    public String getModelFile() {
        return this.MODEL_FILE;
    }

    /**
     *
     * @param modelFile
     */
    public void setModelFile(String modelFile){
        switch (modelFile) {
            case "xception fp16":
                IS_INT8 = false;
                MODEL_FILE = FP16_MODEL_FILE;
                break;
            case "xception int8":
                IS_INT8 = true;
                MODEL_FILE = INT8_MODEL_FILE;
                break;
            default:
                Log.i("tfliteSupport", "Only XceptionFp16/XceptionInt8 can be load!");
        }
    }

    /**
     * ???????????????, ???????????? addNNApiDelegate(), addGPUDelegate()????????????????????????
     *
     * @param activity
     */
    public void initialModel(Context activity) {
        // Initialise the model
        try {

            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);
//            if(IS_INT8){
//                tfliteMeta = new MetadataExtractor(tfliteModel);

//                inputQuantParams = tfliteMeta.getInputTensorQuantizationParams(QUANT_MODEL_INPUT_TENSOR_INDEX);
//                outputQuantParams = tfliteMeta.getInputTensorQuantizationParams(QUANT_MODEL_OUTPUT_TENSOR_INDEX);
//                Log.i("tfliteSupport", "meta "+inputQuantParams.getScale());
//            }
            Log.i("tfliteSupport", "Success reading model: " + MODEL_FILE);

            associatedAxisLabels = FileUtil.loadLabels(activity, LABEL_FILE);
            Log.i("tfliteSupport", "Success reading label: " + LABEL_FILE);

        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * ??????????????????????????????
     *
     * @param bitmap
     * @return LabelInfo
     */
    public LabelInfo recognize(Bitmap bitmap) {

        // xception-tflite????????????:[1, 224, 224,3]
        ImageProcessor imageProcessor;
        TensorImage xceptionTfliteInput;
        if(IS_INT8){
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(0, 255))
                            .add(new QuantizeOp(inputQuantParams.getZeroPoint(), inputQuantParams.getScale()))
                            .add(new CastOp(DataType.UINT8))
                            .build();
            xceptionTfliteInput = new TensorImage(DataType.UINT8);
        } else {
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(0, 255))
                            .build();
            xceptionTfliteInput = new TensorImage(DataType.FLOAT32);
        }
        // ???????????????, ?????????load bitmap?????????,??????????????????????????????bitmap?????????, ?????????????????????????????????????????????bitmap?????????????????????
        // ?????????input??????getBuffer, ?????????bitmap????????????????????????????????????????????????, ????????????load???bitmap?????????float???????????????
        // ?????????????????????process??????????????????uint8??????
        xceptionTfliteInput.load(bitmap);
        xceptionTfliteInput = imageProcessor.process(xceptionTfliteInput);


        // xception-tflite????????????:[1, 200], ?????????v5???GitHub release???????????????tflite??????, ?????????[0,1], ?????????320.
        TensorBuffer probabilityBuffer;
        if(IS_INT8){
            probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.UINT8);
        }else{
            probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);
        }

        // ????????????
        if (null != tflite) {
            // ??????tflite??????????????????batch=1?????????
            tflite.run(xceptionTfliteInput.getBuffer(), probabilityBuffer.getBuffer());
        }

        // ??????????????????????????????
        float[] recognitionArray = probabilityBuffer.getFloatArray();

        // ??????score?????????labelid
        int labelId = 0;
        float maxLabelScores = 0.f;
        for (int j = 0; j < recognitionArray.length; j++) {
            if (recognitionArray[j] > maxLabelScores) {
                maxLabelScores = recognitionArray[j];
                labelId = j;
            }
        }
        return new LabelInfo(labelId, associatedAxisLabels.get(labelId), maxLabelScores);
    }

    /**
     * ???????????????????????????
     *
     * @param recognitions
     * @param bitmap
     * @return
     */
    public ArrayList<Recognition> recognizeBatch(ArrayList<Recognition> recognitions, Bitmap bitmap) {
        ArrayList<Recognition> recognizeBatchResult = new ArrayList<Recognition>();
        for (Recognition r : recognitions) {
            if (r.getLabelName().equals("cat") || r.getLabelName().equals("dog")) {
                int xmin = (int) r.getLocation().left;
                int ymin = (int) r.getLocation().top;
                int xmax = (int) r.getLocation().right;
                int ymax = (int) r.getLocation().bottom;

                Matrix fullScreenTransform = imageProcess.getTransformationMatrix(
                        xmax - xmin, ymax - ymin,
                        INPNUT_SIZE.getWidth(), INPNUT_SIZE.getHeight(),
                        0, false
                );

                Bitmap cropResizeImageBitmap = Bitmap.createBitmap(
                        bitmap, xmin, ymin,
                        (xmax - xmin), (ymax - ymin),
                        fullScreenTransform, false
                );

                try {
                    LabelInfo labelInfo = recognize(cropResizeImageBitmap);
                    if(!labelInfo.labelName.equals("")){
                        r.setLabelId(labelInfo.labelId);
                        r.setLabelName(labelInfo.labelName);
                        r.setLabelScore(labelInfo.labelScore);
                        recognizeBatchResult.add(r);
                    }
                    Log.i("tfliteSupport", this.getModelFile() +" recognize success: "+labelInfo.labelName);
                } catch (Exception e) {
                    Log.e("tfliteSupport", "recognize fail "+e.getMessage());
                }

            }
        }

        return recognizeBatchResult;
    }

    /**
     * ??????NNapi??????
     */
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
        }
    }

    /**
     * ??????GPU??????
     */
    public void addGPUDelegate() {
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
    }

    /**
     * ???????????????
     *
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}

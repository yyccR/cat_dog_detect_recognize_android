package com.example.cat_dog_recognize_lite.recognizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import com.example.cat_dog_recognize_lite.utils.ImageProcess;
import com.example.cat_dog_recognize_lite.utils.Recognition;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
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

    private final Size INPNUT_SIZE = new Size(224, 224);
    private final int[] OUTPUT_SIZE = new int[]{1, 200};
    private Boolean IS_INT8 = false;
    private final float THRESHOLD = 0.25f;
    ImageProcess imageProcess = new ImageProcess();

    private final String LABEL_FILE = "cat_dog_breeds_cn.txt";
    private String FP16_MODEL_FILE = "cat_dog_classify_xception_0322_fp16.tflite";
    private String INT8_MODEL_FILE = "cat_dog_classify_xception_0322_int8.tflite";
    private String MODEL_FILE;

    // 如果替换自己的int8 quant模型, 需要修改input/output Tensor的索引
    private int QUANT_MODEL_INPUT_TENSOR_INDEX = 231;
    private int QUANT_MODEL_OUTPUT_TENSOR_INDEX = 232;
    MetadataExtractor.QuantizationParams inputQuantParams;
    MetadataExtractor.QuantizationParams outputQuantParams;
    private Interpreter tflite;
    private MetadataExtractor tfliteMeta;
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
                MODEL_FILE = INT8_MODEL_FILE;
                IS_INT8 = true;
                break;
            default:
                Log.i("tfliteSupport", "Only XceptionFp16/XceptionInt8 can be load!");
        }
    }

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     *
     * @param activity
     */
    public void initialModel(Context activity) {
        // Initialise the model
        try {

            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);
            if(IS_INT8){
                MetadataExtractor tfliteMeta = new MetadataExtractor(tfliteModel);
                inputQuantParams = tfliteMeta.getInputTensorQuantizationParams(QUANT_MODEL_INPUT_TENSOR_INDEX);
                outputQuantParams = tfliteMeta.getInputTensorQuantizationParams(QUANT_MODEL_OUTPUT_TENSOR_INDEX);
            }
            Log.i("tfliteSupport", "Success reading model: " + MODEL_FILE);

            associatedAxisLabels = FileUtil.loadLabels(activity, LABEL_FILE);
            Log.i("tfliteSupport", "Success reading label: " + LABEL_FILE);

        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * 识别检测框内猫狗类别
     *
     * @param bitmap
     * @return LabelInfo
     */
    public LabelInfo recognize(Bitmap bitmap) {

        // xception-tflite的输入是:[1, 224, 224,3]
        ImageProcessor imageProcessor;
        TensorImage xceptionTfliteInput;
        if(IS_INT8){
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(0, 255))
                            .add(new QuantizeOp(inputQuantParams.getZeroPoint(), inputQuantParams.getScale()))
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
        xceptionTfliteInput.load(bitmap);
        xceptionTfliteInput = imageProcessor.process(xceptionTfliteInput);


        // xception-tflite的输出是:[1, 200], 可以从v5的GitHub release处找到相关tflite模型, 输出是[0,1], 处理到320.
        TensorBuffer probabilityBuffer;
        if(IS_INT8){
            probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.UINT8);
        }else{
            probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);
        }


        // 推理计算
        if (null != tflite) {
            // 这里tflite默认会加一个batch=1的纬度
            tflite.run(xceptionTfliteInput.getBuffer(), probabilityBuffer.getBuffer());
        }

        // 输出数据被平铺了出来
        float[] recognitionArray = probabilityBuffer.getFloatArray();

        // 找到score最大的labelid
        int labelId = 0;
        float maxLabelScores = 0.f;
        for (int j = 0; j < recognitionArray.length; j++) {
            if (recognitionArray[j] > maxLabelScores) {
                maxLabelScores = recognitionArray[j];
                labelId = j;
            }
        }
        Log.i("recognize", "label id: "+labelId+" label name: "+associatedAxisLabels.toString() + " label score: "+maxLabelScores);
        return new LabelInfo(labelId, associatedAxisLabels.get(labelId), maxLabelScores);
    }

    /**
     * 批量识别对应的类别
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

                LabelInfo labelInfo = recognize(cropResizeImageBitmap);
                if(!labelInfo.labelName.equals("")){
                    r.setLabelId(labelInfo.labelId);
                    r.setLabelName(labelInfo.labelName);
                    r.setLabelScore(labelInfo.labelScore);
                    recognizeBatchResult.add(r);
                }
            }
        }

        return recognizeBatchResult;
    }

    /**
     * 添加NNapi代理
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
     * 添加GPU代理
     */
    public void addGPUDelegate() {
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
    }

    /**
     * 添加线程数
     *
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}

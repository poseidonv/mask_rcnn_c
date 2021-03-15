#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <assert.h>
#include <vector>
#include <map>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <torchvision/nms.h>
// #include <torchvision/csrc/cuda/vision_cuda.h>


using namespace std;

class Transformer
{
public:
    Transformer(){};

    void get_transformer(cv::Mat &original_image);

    void apply_image(cv::Mat &original_image);
private:
    int short_edge_length = 800;
    int max_size = 1333;
    int w, h;
    float neww, newh;
    int interp = cv::INTER_LINEAR;
};

void Transformer::get_transformer(cv::Mat &original_image)
{
    h = original_image.rows;
    w = original_image.cols;
    int size = rand() % 1 + 800;

    float scale = size * 1.0 / min(h, w);
    if (h < w)
    {
        newh= size;
        neww = w * scale;
    }
    else
    {
        newh = h * scale;
        neww = size;
    }

    if (max(newh, neww) > max_size)
    {
        scale = max_size / max(newh, neww);
        newh = newh * scale;
        neww = neww * scale;
    }
    neww = int(neww + 0.5);
    newh = int(newh + 0.5);
}

void Transformer::apply_image(cv::Mat &original_image)
{
    assert(original_image.rows == h && original_image.cols == w);
    assert(original_image.dims <= 4);

    cv::resize(original_image, original_image, cv::Size(neww, newh));
}

class MaskRcnn
{
public:
    MaskRcnn(std::string model_path);

    void Run();

    bool SetImage(cv::Mat& image);

    bool ProcessImage(cv::Mat& image);

    at::Tensor ImageToTensor(cv::Mat& processed_image);

    c10::IValue Inference(at::Tensor& tensor_image);

    vector<cv::Mat> ProcessOutput(c10::IValue& output);

    at::Tensor ProcessMaskProbs(at::Tensor bbox, at::Tensor mask_probs);

    cv::Mat CombineMask(cv::Mat boxes, at::Tensor img_masks);
private:
    torch::jit::script::Module module;
    int height = 480;
    int width = 640;
    c10::Device mdevice = at::kCPU;
    cv::Mat mimage;
    cv::Mat mprocessed_image;

};

MaskRcnn::MaskRcnn(std::string model_path)
{
    cout<<model_path<<endl;
    bool cuda_availible = torch::cuda::is_available();
    std::cout<<"cuda availible: "<<cuda_availible<<std::endl;
    module = torch::jit::load(model_path);

    assert(module.buffers().size() > 0);
    cout<<(*begin(module.buffers())).device()<<endl;
    mdevice = (*begin(module.buffers())).device();
    cout<<mdevice<<endl;
}

void MaskRcnn::Run()
{
    bool flag = ProcessImage(mimage);
    at::Tensor tensor_image = ImageToTensor(mprocessed_image);
    c10::IValue output = Inference(tensor_image);
    ProcessOutput(output);
}

bool MaskRcnn::SetImage(cv::Mat& image)
{
    mimage = image.clone();
}

bool MaskRcnn::ProcessImage(cv::Mat& image)
{
    cv::resize(image, mprocessed_image, cv::Size(640,480));
    const int height = mprocessed_image.rows;
    const int width = mprocessed_image.cols;

    assert( height % 32 == 0 && width % 32 == 0);

    if( mprocessed_image.channels() == 4)
    {
        cv::cvtColor(mprocessed_image, mprocessed_image, cv::COLOR_BGRA2BGR);
    }

    return true;
}

at::Tensor MaskRcnn::ImageToTensor(cv::Mat& processed_image)
{
    at::Tensor input = torch::from_blob(processed_image.data, {height, width, 3}, torch::kUInt8);
    cout<<input.sizes()<<endl;
    input = input.to(mdevice, torch::kFloat).permute({2,0,1}).contiguous();
    
    return input;
}

c10::IValue MaskRcnn::Inference(at::Tensor& tensor_image)
{
    c10::IValue output = module.forward({tensor_image});

    return output;
}

vector<cv::Mat> MaskRcnn::ProcessOutput(c10::IValue& output)
{
    vector<cv::Mat> vOutput;

    auto outputs = output.toTuple()->elements();
    at::Tensor bbox = outputs[0].toTensor().to(at::kCPU), scores = outputs[1].toTensor().to(at::kCPU),
       labels = outputs[2].toTensor().to(at::kCPU), mask_probs = outputs[3].toTensor().to(at::kCPU);

    cv::Mat scores_mat = cv::Mat(scores.sizes()[0],1,CV_32FC1, cv::Scalar(0));
    std::memcpy((void *) scores_mat.data, scores.data_ptr<float>(), 4*scores.numel());
    // cv::sortIdx(score_mat, score_mat, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
    // cout<<"scores: "<<scores_mat<<endl;

    cv::Mat labelsmat = cv::Mat(labels.size(0), 1,CV_64FC1, cv::Scalar(0));
    std::memcpy((void *) labelsmat.data, labels.data_ptr<long>(), 8*labels.numel());
    cout<<"labels: "<<endl;
    for(int i = 0; i<labelsmat.rows; i++)
    {
        cout<<int(labelsmat.at<long>(i,0))<<endl;
    }

    cv::Mat boxes = cv::Mat(bbox.size(0), 4, CV_32FC1, cv::Scalar(0));
    // cout<<sizeof(bbox[0][0])<<endl;
    std::memcpy((void *) boxes.data, bbox.data_ptr<float>(), 4*bbox.numel());
    // cout<<"boxes: "<<boxes<<endl;

    at::Tensor img_masks = ProcessMaskProbs(bbox, mask_probs);
    cv::Mat mask = CombineMask(boxes, img_masks);
    cv::resize(mask, mask, cv::Size(mimage.cols, mimage.rows));
    cv::imwrite("mask.jpg", mask);


    vOutput.push_back(scores_mat);
    vOutput.push_back(labelsmat);
    vOutput.push_back(boxes);
    vOutput.push_back(mask);
    return vOutput;
    
}

at::Tensor MaskRcnn::ProcessMaskProbs(at::Tensor bbox, at::Tensor mask_probs)
{
    int n = bbox.sizes()[0];
    float scale_x, scale_y;
    scale_x = mprocessed_image.cols * 1.0 / mprocessed_image.cols;
    scale_y = mprocessed_image.rows * 1.0 / mprocessed_image.rows;
    at::Tensor scale = torch::tensor({scale_x, scale_y, scale_x, scale_y});
    vector<at::Tensor> vbbox = bbox.split(1,1);
    vbbox[0] = vbbox[0].mul(scale_x).clamp(0,mprocessed_image.cols);
    vbbox[1] = vbbox[1].mul(scale_y).clamp(0,mprocessed_image.rows);
    vbbox[2] = vbbox[2].mul(scale_x).clamp(0,mprocessed_image.cols);
    vbbox[3] = vbbox[3].mul(scale_y).clamp(0,mprocessed_image.rows);
    at::Tensor subw = (vbbox[2] - vbbox[0]).greater(0).expand({n,4});
    at::Tensor subh = (vbbox[3] - vbbox[1]).greater(0).expand({n,4});
    at::Tensor areas = (vbbox[2] - vbbox[0])*(vbbox[3] - vbbox[1]);
    // cout<<"areas: "<<endl<<areas<<endl;
    bbox = torch::cat(vbbox, 1);
    bbox = bbox.index(subw).reshape({-1,4}).index(subh).reshape({-1,4});
    // cout<<"bbox[0]: "<<bbox[0]<<endl;
    // cout<<"bbox[1]: "<<bbox[1]<<endl;

    int x0,y0,x1,y1;
    x0 = 0;
    y0 = 0;
    x1 = mprocessed_image.cols;
    y1 = mprocessed_image.rows;
    // cout<<"mask probs: "<<mask_probs[0][0]<<endl;
    at::Tensor img_x = torch::arange(x0, x1).to(at::kFloat) + 0.5;
    at::Tensor img_y = torch::arange(y0, y1).to(at::kFloat) + 0.5;
    img_x = (img_x - vbbox[0])/(vbbox[2] - vbbox[0])*2 - 1;
    img_y = (img_y - vbbox[1])/(vbbox[3] - vbbox[1])*2 - 1;
    at::Tensor gx = img_x.unsqueeze(1).expand({img_x.sizes()[0], img_y.sizes()[1], img_x.sizes()[1]});
    at::Tensor gy = img_y.unsqueeze(2).expand({img_x.sizes()[0], img_y.sizes()[1], img_x.sizes()[1]});
    at::Tensor grid = torch::stack({gx, gy}, 3);
    // cout<<"grid type: "<<grid.toString()<<endl;
    // cout<<"mask prob type: "<<mask_probs.toString()<<endl;
    at::Tensor img_masks = torch::grid_sampler(mask_probs, grid, 0, 0, false);
    img_masks = img_masks.squeeze(1);
    // cout<<img_masks.sizes()<<endl;
    // cout<<img_masks[0][400]<<endl;
    img_masks = img_masks.greater(0.5);

    return img_masks;
}

cv::Mat MaskRcnn::CombineMask(cv::Mat boxes,at::Tensor img_masks)
{
    cv::Mat mask = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    cv::Mat mask0 = cv::Mat(480, 640, CV_8UC1, cv::Scalar(0));

    vector<cv::Scalar> colors = {cv::Scalar(0,64,0),cv::Scalar(0,0,64), cv::Scalar(0,64,64), cv::Scalar(64,0,64),
                                cv::Scalar(0,128,64),cv::Scalar(128,0,64), cv::Scalar(64,128,0), cv::Scalar(64,128,128),
                                cv::Scalar(256,0,64),cv::Scalar(128,256,64),cv::Scalar(256,64,128)};
    map<int, string> classname;
    classname[0] = "person";
    classname[56] = "chair";
    classname[62] = "tv";
    classname[66] = "keyboard";
    classname[64] = "mouse";
    classname[39] = "bottle";
    int img_area = mprocessed_image.cols*mprocessed_image.rows;
    for(int i=0; i<boxes.rows; i++)
    {
        std::memcpy((void *) mask0.data, img_masks[i].data_ptr<bool>(), 1*img_masks[i].numel());
        mask.setTo(colors[i], mask0);
        int x0,y0,x1,y1;
        x0 = int(boxes.at<float>(i,0));
        y0 = int(boxes.at<float>(i,1));
        x1 = int(boxes.at<float>(i,2));
        y1 = int(boxes.at<float>(i,3));
        int area = (x1 - x0)*(y1 - y0);
        float font_size = area * 2.27 /img_area;
        if(font_size < 0.1) font_size = 0.1;

        cv::rectangle(mask, cv::Rect(cv::Point2i(x0, y0), cv::Point2i(x1, y1)) ,cv::Scalar(255,255,255));
        if(mask.at<cv::Vec3b>(300,600) == cv::Vec3b(0,0,64))
            cout<<"Person!"<<endl;
        // int classnum = int(labelsmat.at<long>(i,0));
        // cout<<classnum<<endl;
        // cv::putText(mask, classname[classnum], cv::Point2i(x0, int(y0+font_size*10)), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(128,128,128));
    }

    return mask;
}

// experimental.
int main(int argc, const char** argv) {
    // arguments:
    //   model.ts input.jpg
    //if (argc != 3) {
        //return 1;
    //}
    // std::string image_file = argv[2];
    // at::Tensor x = torch::ones({3,3});
    // cout<<x<<endl;
    // at::Tensor y = torch::tensor({{10,10,20,20}, {15,15,25,25}, {20,20,30,30}}).to(at::kFloat);
    // int n = y.sizes()[0];
    // // at::Tensor z = torch::tensor({{10,10,20,20}, {0,0,0,0}, {20,20,30,30}});
    // // at::Tensor z = torch::tensor({{1,1,1,1}});
    // vector<at::Tensor> vy = y.split(1, 1);
    // at::Tensor subw = (vy[2] - vy[0]).greater(0).expand({n,4});
    // at::Tensor subh = (vy[3] - vy[1]).greater(0).expand({n,4});
    // at::Tensor a = torch::tensor({{1},{0},{1}}).to(at::kBool).expand({3,4});
    // // cout<<a<<endl;
    // // cout<<subw.sizes()<<endl;
    // // cout<<subh<<endl;
    // // cout<<y.where(a, z)<<endl;
    // cout<<y.index(subw).reshape({-1,4}).index(subh).reshape({-1,4})<<endl;
    // // for(int i = 0; i < y.sizes()[0]; i++ )
    // // {
    // //     if(subw[i].data_ptr() && subh[i].data_ptr())
    // //     {
            
    // //     }
    // // }
    // // cout<<y.is_nonzero().sizes()<<endl;
    // // cout<<y.index_select(y.nonzero())<<endl;
    // // cout<<y..index(y.is_nonzero())<<endl;
    // // cout<<y.sizes()<<endl;
    // // vector<at::Tensor> vy = y.split(3, 1);
    // vy[0] = vy[0].mul(2.0).clamp(5,30);
    // cout<<vy[0]<<endl;
    // vy[1] = vy[1].clamp(5,30);
    // cout<<vy[1]<<endl;
    // // cout<<vy[2]<<endl;
    // y = torch::cat(vy,1);
    // cout<<y<<endl;

    // at::Tensor img_y = torch::arange(0,10);
    // at::Tensor y0 = torch::tensor({{1}, {2}, {3}});
    // at::Tensor y1 = torch::tensor({{3}, {4}, {5}});
    // at::Tensor gy = ((img_y - y0)/(y1 - y0)*2 - 1).unsqueeze(1);
    // // cout<<((img_y - y0)/(y1 - y0)*2 - 1).unsqueeze(1).sizes()<<endl;
    // at::Tensor grid = torch::stack({gy,gy}, 3);
    // cout<<grid.sizes()<<endl;
    // exit(-1);

    // cout<<y<<endl;
    // cout<<(x.add_(y))<<endl;
    ///home/poseidon/Documents/github/detectron2-master/output  /home/poseidon/Documents/mask_rcnn_c/model_cuda_no_aug.ts
    ///home/poseidon/Documents/00/image_0/000000.png
    MaskRcnn maskRcnn("/home/poseidon/Documents/github/detectron2-master/output/model.ts");
    vector<string> filenames = {"1341846313.654184.png","1341846313.553992.png","1341846313.592026.png","1341846313.622103.png","1341846313.686156.png","1341846313.654184.png"};
    string filedir = "/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/";
    for(int i = 0;i<filenames.size(); i++)
    {
        filenames[i] = filedir + filenames[i];
        cout<<filenames[i]<<endl;
        cv::Mat image = cv::imread("/home/poseidon/Documents/00/image_0/000000.png", cv::ImreadModes::IMREAD_COLOR);
        maskRcnn.SetImage(image);
        maskRcnn.Run();
        exit(-1);
    }
    // cv::Mat image = cv::imread("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.686156.png", cv::ImreadModes::IMREAD_COLOR);
    // maskRcnn.SetImage(image);
    // maskRcnn.Run();
//--------------------------------------------------------
    // bool cuda_availible = torch::cuda::is_available();
    // std::cout<<"cuda availible: "<<cuda_availible<<std::endl;
    // torch::DeviceType device_type;
    // device_type = torch::kCUDA;

    // // torch::autograd::AutoGradMode guard(false);
    // auto module = torch::jit::load("/home/poseidon/Documents/mask_rcnn_c/model_cuda_no_aug.ts");
    // // module.to(device_type);

    // assert(module.buffers().size() > 0);
    // cout<<(*begin(module.buffers())).device()<<endl;
    // // Assume that the entire model is on the same device.
    // // We just put input to this device.
    // auto device = (*begin(module.buffers())).device();
    // string img_path = argv[1];
    // // cv::Mat original_image = cv::imread("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png", cv::ImreadModes::IMREAD_COLOR);
    // cv::Mat original_image_1 = cv::imread(img_path, cv::ImreadModes::IMREAD_COLOR);
    // cv::Mat original_image;
    // cv::resize(original_image_1, original_image, cv::Size(640,480));
    // // Transformer transformer;
    // // transformer.get_transformer(input_image);
    // // transformer.apply_image(input_image);
    // cv::Mat input_image = original_image.clone();
    // cout<<"image channels: "<<input_image.channels()<<endl;
    // const int height = input_image.rows;
    // const int width = input_image.cols;
    // if( input_image.channels() == 4)
    // {
    //     cv::cvtColor(input_image, input_image, cv::COLOR_BGRA2BGR);
    //  }

    // // assert( height % 32 == 0 && width % 32 == 0);

    // const int channels = 3;

    // // auto input = torch::from_blob(input_image.data, {1, height, width, channels}, torch::kUInt8);
    // auto input = torch::from_blob(input_image.data, {height, width, channels}, torch::kUInt8);

    // // input = input.to(device, torch::kFloat).permute({0, 3, 1, 2}).contiguous();
    // input = input.to(device, torch::kFloat).permute({2,0,1}).contiguous();

    // std::array<float, 3> im_info_data{height*1.0f, width*1.0f, 1.0f};

    // auto im_info = torch::from_blob(im_info_data.data(), {1, 3}).to(device);
    // clock_t start, end;
    // // auto output = module.forward({std::make_tuple(input, im_info)});
    // start = clock();
    // c10::IValue output = module.forward({input});
    // end = clock();
    // std::cout<<"1 forward total use time : "<<(double)(end - start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    // if(device.is_cuda())
    //     c10::cuda::getCurrentCUDAStream().synchronize();
    // clock_t start1 = clock();
    // // A = segnet.predict("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png");
    // start = clock();
    // output = module.forward({input});
    // end = clock();
    // std::cout<<"2 forward total use time : "<<(double)(end - start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    // int N_benchmark = 3;
    // auto start_time = chrono::high_resolution_clock::now();
    // for (int i = 0; i < N_benchmark; ++i) {
    //     // output = module.forward({std::make_tuple(input, im_info)});
    //     start = clock();
    //     output = module.forward({input});
    //     end = clock();
    //     std::cout<<i<<" forward total use time : "<<(double)(end - start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    //     cout<<"is cuda: "<<device.is_cuda()<<endl;
    //     if (device.is_cuda())
    //         c10::cuda::getCurrentCUDAStream().synchronize();
    // }
    // auto end_time = chrono::high_resolution_clock::now();
    // auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time) .count();
    // cout << "Latency (should vary with different inputs): "<< ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;
    // clock_t end1 = clock();
    // std::cout<<"2------------total use time : "<<(double)(end1 - start1)/CLOCKS_PER_SEC<<"s"<<std::endl;//cpu:4.5s cuda:0.15s

    // auto outputs = output.toTuple()->elements();
    // // parse Mask R-CNN outputs
    // auto bbox = outputs[0].toTensor().to(at::kCPU), scores = outputs[1].toTensor().to(at::kCPU),
    //    labels = outputs[2].toTensor().to(at::kCPU), mask_probs = outputs[3].toTensor().to(at::kCPU);
    // cv::Mat score_mat = cv::Mat(41,1,CV_32FC1, cv::Scalar(0));
    // std::memcpy((void *) score_mat.data, scores.data_ptr<float>(), 4*scores.numel());
    // cv::sortIdx(score_mat, score_mat, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
    // cout << "bbox: " << bbox.toString() << " " << bbox.sizes() << endl;
    // cout << "scores: " << scores.toString() << " " << scores.sizes() << endl;
    // cout << "labels: " << labels.toString() << " " << labels.sizes() << endl;
    // cout << "mask_probs: " << mask_probs.toString() << " " << mask_probs.sizes()<< endl;
    // // cout<< "labels "<<labels<<endl;
    // // cout<<"scores: "<<scores<<endl;
    // // cout<<"scores: "<<score_mat<<endl;
    // // cout<<"mask_probs: "<<mask_probs[0][0]<<endl;


    // int n = bbox.sizes()[0];
    // float scale_x, scale_y;
    // scale_x = original_image.cols * 1.0 / input_image.cols;
    // scale_y = original_image.rows * 1.0 / input_image.rows;
    // // cout<<bbox[0]<<endl;
    // at::Tensor scale = torch::tensor({scale_x, scale_y, scale_x, scale_y});
    // // cout<<"scale: "<<scale<<endl;
    // // bbox = bbox.add(scale);
    // // cout<<bbox[0]<<endl;
    // // cout<<bbox[1]<<endl;
    // vector<at::Tensor> vbbox = bbox.split(1,1);
    // vbbox[0] = vbbox[0].mul(scale_x).clamp(0,original_image.cols);
    // vbbox[1] = vbbox[1].mul(scale_y).clamp(0,original_image.rows);
    // vbbox[2] = vbbox[2].mul(scale_x).clamp(0,original_image.cols);
    // vbbox[3] = vbbox[3].mul(scale_y).clamp(0,original_image.rows);
    // at::Tensor subw = (vbbox[2] - vbbox[0]).greater(0).expand({n,4});
    // at::Tensor subh = (vbbox[3] - vbbox[1]).greater(0).expand({n,4});
    // at::Tensor areas = (vbbox[2] - vbbox[0])*(vbbox[3] - vbbox[1]);
    // cout<<"areas: "<<endl<<areas<<endl;
    // // cout<<bbox.nonzero()<<endl;
    // bbox = torch::cat(vbbox, 1);
    // bbox = bbox.index(subw).reshape({-1,4}).index(subh).reshape({-1,4});
    // // cout<<bbox[0]<<endl;
    // // cout<<bbox[1]<<endl;

    // int x0,y0,x1,y1;
    // x0 = 0;
    // y0 = 0;
    // x1 = original_image.cols;
    // y1 = original_image.rows;
    // // cout<<"mask probs: "<<mask_probs[0][0]<<endl;
    // at::Tensor img_x = torch::arange(x0, x1).to(at::kFloat) + 0.5;
    // at::Tensor img_y = torch::arange(y0, y1).to(at::kFloat) + 0.5;
    // img_x = (img_x - vbbox[0])/(vbbox[2] - vbbox[0])*2 - 1;
    // img_y = (img_y - vbbox[1])/(vbbox[3] - vbbox[1])*2 - 1;
    // at::Tensor gx = img_x.unsqueeze(1).expand({img_x.sizes()[0], img_y.sizes()[1], img_x.sizes()[1]});
    // at::Tensor gy = img_y.unsqueeze(2).expand({img_x.sizes()[0], img_y.sizes()[1], img_x.sizes()[1]});
    // at::Tensor grid = torch::stack({gx, gy}, 3);
    // cout<<"grid type: "<<grid.toString()<<endl;
    // cout<<"mask prob type: "<<mask_probs.toString()<<endl;
    // at::Tensor img_masks = torch::grid_sampler(mask_probs, grid, 0, 0, false);
    // img_masks = img_masks.squeeze(1);
    // // cout<<img_masks.sizes()<<endl;
    // // cout<<img_masks[0][400]<<endl;
    // img_masks = img_masks.greater(0.5);

    // // cout<<gx.sizes()<<endl;
    // // cout<<gy.sizes()<<endl;
    // cv::Mat mask = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    // cv::Mat mask0 = cv::Mat(480, 640, CV_8UC1, cv::Scalar(0));
    // // std::memcpy((void *) mask0.data, img_masks[0].data_ptr<bool>(), 1*img_masks[0].numel());
    // cv::Mat boxes = cv::Mat(bbox.size(0), 4, CV_32FC1, cv::Scalar(0));
    // cv::Mat labelsmat = cv::Mat(labels.size(0), 1,CV_64FC1, cv::Scalar(0));
    // std::memcpy((void *) labelsmat.data, labels.data_ptr<long>(), 8*labels.numel());
    // // cout<<sizeof(bbox[0][0])<<endl;
    // cout<<bbox<<endl;
    // std::memcpy((void *) boxes.data, bbox.data_ptr<float>(), 4*bbox.numel());
    // vector<cv::Scalar> colors = {cv::Scalar(0,64,0),cv::Scalar(0,0,64), cv::Scalar(0,64,64), cv::Scalar(64,0,64),
    //                             cv::Scalar(0,128,64),cv::Scalar(128,0,64), cv::Scalar(64,128,0), cv::Scalar(64,128,128),
    //                             cv::Scalar(256,0,64),cv::Scalar(128,256,64),cv::Scalar(256,64,128)};
    // map<int, string> classname;
    // classname[0] = "person";
    // classname[56] = "chair";
    // classname[62] = "tv";
    // classname[66] = "keyboard";
    // classname[64] = "mouse";
    // classname[39] = "bottle";
    // int img_area = original_image.cols*original_image.rows;
    // for(int i=0; i<bbox.size(0); i++)
    // {
    //     std::memcpy((void *) mask0.data, img_masks[i].data_ptr<bool>(), 1*img_masks[i].numel());
    //     mask.setTo(colors[i], mask0);
    //     int x0,y0,x1,y1;
    //     x0 = int(boxes.at<float>(i,0));
    //     y0 = int(boxes.at<float>(i,1));
    //     x1 = int(boxes.at<float>(i,2));
    //     y1 = int(boxes.at<float>(i,3));
    //     int area = (x1 - x0)*(y1 - y0);
    //     float font_size = area * 2.27 /img_area;
    //     if(font_size < 0.1) font_size = 0.1;

    //     cv::rectangle(mask, cv::Rect(cv::Point2i(x0, y0), cv::Point2i(x1, y1)) ,cv::Scalar(255,255,255));
    //     int classnum = int(labelsmat.at<long>(i,0));
    //     cout<<classnum<<endl;
    //     cv::putText(mask, classname[classnum], cv::Point2i(x0, int(y0+font_size*10)), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(128,128,128));
    // }
    // // cout<<boxes<<endl;
    // cv::resize(mask, mask, cv::Size(original_image_1.cols, original_image_1.rows));
    // cv::imwrite("mask0.jpg", mask);
    
    // int num_instances = bbox.sizes()[0];
    // cout<<"num_instances "<<num_instances<<endl;
    //--------------------------------------------------------


    return 0;
}


// 0,  0, 56, 62, 62, 64, 39, 56, 66, 73
//([0.9993, 0.9990, 0.9973, 0.9963, 0.9926, 0.9820, 0.9539, 0.9404, 0.9201,0.7070
// 0, 0, 56, 62, 62, 66, 56, , 6664, 39

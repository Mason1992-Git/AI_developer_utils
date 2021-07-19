//
// Created by Administrator on 2021/7/19 0019.
//

#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>

using namespace std;

vector<float> iou(vector<float> box, vector<vector<float>> boxes) {
    float box_aera = box[3] * box[4];
    int num = boxes.size();

    vector<float> boxes_aera;
    for (int i = 0; i < num; i++) {
        float aera = boxes[i][3] * boxes[i][4];
        boxes_aera.push_back(aera);
    }
    /*
    x1 = torch::max((box[1] - box[3] / 2), (boxes[:,1]-boxes[:,3]/2))
    y1 = torch::max((box[2] - box[4] / 2), (boxes[:,2]-boxes[:,4]/2))
    x2 = torch::min((box[1] + box[3] / 2), (boxes[:,1]+boxes[:,3]/2))
    y2 = torch::min((box[2] + box[4] / 2), (boxes[:,2]+boxes[:,4]/2))*/

    return boxes_aera;
}


int main() {
    vector<float> box = {0.8, 0, 0, 40, 40};
    vector<vector<float>> boxes = {{0.85, 0,  0,  40, 40},
                                   {0.6,  20, 20, 60, 60},
                                   {0.9,  10, 25, 80, 90}};
    vector<float> cal_iou = iou(box, boxes);


    cout << "vector_size = " << boxes.size() << endl;
    int num = boxes.size();
    for (int i = 0; i < num; i++) {
        cout << "iou = " << cal_iou[i] << endl;
    }

}

/*
def iou(box, boxes, mode="inter"):
box_area = box[3] * box[4]
boxes_area = (boxes[:, 3] ) * (boxes[:, 4])
# box_area = (box[3] - box[1]) * (box[4] - box[2])
# boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])


x1 = torch.max((box[1]-box[3]/2), (boxes[:,1]-boxes[:,3]/2))
y1 = torch.max((box[2]-box[4]/2), (boxes[:,2]-boxes[:,4]/2))
x2 = torch.min((box[1]+box[3]/2), (boxes[:,1]+boxes[:,3]/2))
y2 = torch.min((box[2]+box[4]/2), (boxes[:,2]+boxes[:,4]/2))
#
    # x1 = torch.min(box[1], boxes[:, 1])
# y1 = torch.min(box[2], boxes[:, 2])
# x2 = torch.max(box[3], boxes[:, 3])
# y2 = torch.max(box[4], boxes[:, 4])

w = torch.clamp(x2 - x1, min=0)
# print("x2:",x2)
# print("x1:",x1)
# print(x2-x1)
# print("w:",w)
h = torch.clamp(y2 - y1, min=0)

inter = w * h

if mode == 'inter':
return inter / (box_area + boxes_area - inter)
elif mode == 'min':
return inter / torch.min(box_area, boxes_area)*/

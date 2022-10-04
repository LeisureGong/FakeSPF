//
// Created by lei on 2022/10/4.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include<vector>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "metis.h"
#include <map>

using namespace std;

// 两个vector求交集
vector<int> vectors_intersection(vector<int> v1, vector<int> v2) {
    vector<int> v;
    sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());
    set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
    return v;
}

double nmi_score(const double* trainLabel, const idx_t* part2, int m) {
    // Mutual information
    double MI = 0.0;
    // 避免log函数中参数为0
    double eps = 1.4e-45;
    unordered_set<int> A_ids;
    unordered_set<int> B_ids;
    for (int i = 0; i < m; i++)
        A_ids.emplace(*(trainLabel + i));
    for (int i = 0; i  < m; i++)
        B_ids.emplace(*(part2 + i));
    map<int, vector<int>> map_A;
    map<int, vector<int>> map_B;
    // 构建<label, index list> 关系
    for (int i = 0; i < m; i++) {
        int key = *(trainLabel + i);
        auto iter = map_A.find(key);
        if (iter != map_A.end()){
            vector<int> vec_tmp = map_A[key];
            vec_tmp.push_back(i);
            map_A[key] = vec_tmp;
        } else {
            vector<int> tmp = {i};
            map_A[key] = tmp;
        }
    }
    for (int i = 0; i < m; i++) {
        int key = *(part2 + i);
        auto iter = map_B.find(key);
        if (iter != map_B.end()) {
            vector<int> vec_tmp = map_B[key];
            vec_tmp.push_back(i);
            map_B[key] = vec_tmp;
        } else {
            vector<int> tmp = {i};
            map_B[key] = tmp;
        }
    }

    for (auto a = A_ids.begin(); a != A_ids.end(); a++) {
        for (auto b = B_ids.begin(); b!= B_ids.end(); b++) {
            // 记录A_ids中和*a相同值的所有index
            vector<int> idx_a_list = map_A[*a];
            vector<int> idx_b_list = map_B[*b];
            // 相同index的位置
            vector<int> idx_ab_list = vectors_intersection(idx_a_list, idx_b_list);
            double px = 1.0 * idx_a_list.size() / m;
            double py = 1.0 * idx_b_list.size() / m;
            double pxy = 1.0 * idx_ab_list.size() / m;
            MI += pxy * (log(pxy / (px * py) + eps) / log(2));
        }
    }
    // Normalized Mutual information
    double Hx = 0.0;
    for (auto a = A_ids.begin(); a != A_ids.end(); a++) {
        double idx_a_cnt = 1.0*map_A[*a].size();
        Hx = Hx - (idx_a_cnt / m) * (log(idx_a_cnt / m + eps) / log(2));
    }
    double Hy = 0;
    for (auto b = B_ids.begin(); b != B_ids.end(); b++) {
        double idx_b_cnt = 1.0 * map_B[*b].size();
        Hy = Hy - (idx_b_cnt / m) * (log(idx_b_cnt / m + eps) / log(2));
    }
    double MIhat = 2.0 * MI / (Hx + Hy);
    return MIhat;
}

int main() {
    double A[17] = {1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3};
    idx_t B[17] = {2,1,3,1,3,1,1,2,3,1,3,1,1,3,3,1,2};
    double score = nmi_score(A, B, 17);
    cout << score << endl;
    return 0;
}
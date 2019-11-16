#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

__global__ 
void hsv_k_mean(int *output_cluster, int K, float *old_cluster, 
                float *hsv_img, int img_height, int img_width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row*img_width+col < img_height*img_width){

    // printf("[%f]\n", hsv_img[row*img_width + col*3 + 0]);
    // printf("[%f]\n", old_cluster[0*3 + 0]);
    
    double min_distance;
    min_distance = sqrt(pow(hsv_img[row*img_width + col*3 + 0] - old_cluster[0*3 + 0], 2)
                       +pow(hsv_img[row*img_width + col*3 + 1] - old_cluster[0*3 + 1], 2)
                       +pow(hsv_img[row*img_width + col*3 + 2] - old_cluster[0*3 + 2], 2));
    int picked_cluster = 0;

    // printf("[%f]\n", min_distance);

    double distance;

    for(int i=1 ; i<K ; i++){
      distance = sqrt(pow(hsv_img[(row*img_width)+(col*3)+0] - old_cluster[i*3+0], 2)
                     +pow(hsv_img[(row*img_width)+(col*3)+1] - old_cluster[i*3+1], 2)
                     +pow(hsv_img[(row*img_width)+(col*3)+2] - old_cluster[i*3+2], 2));
      if(distance < min_distance){
        picked_cluster = i;
        min_distance = distance;
      }
    }

    output_cluster[(row*img_width)+col] = picked_cluster;

    // printf("[%f]\n", output_cluster[row*img_width + col]);

    /*
    sum_cluster[(picked_cluster*3)+0] += hsv_img[(row*img_width)+(col*3)+0];
    sum_cluster[(picked_cluster*3)+1] += hsv_img[(row*img_width)+(col*3)+1];
    sum_cluster[(picked_cluster*3)+2] += hsv_img[(row*img_width)+(col*3)+2];
    */
    
    // printf("[%f]\n", sum_cluster[(picked_cluster*3)+0]);

    // count_cluster[picked_cluster] += 1;
  }
}

int main(){

    // Host Allocate
    // --- K
    int K = 3;

    // --- Height & Width
    int height = 1000;
    int width = 1000;

    // --- Image
    unsigned int size_img = height * width * 3;
    unsigned int mem_size_img = sizeof(float) * size_img;
    float* h_img = (float*) malloc(mem_size_img);
    float* h_seg = (float*) malloc(mem_size_img);

    // --- output
    unsigned int mem_size_output = sizeof(int) * size_img;
    int* h_output = (int*) malloc(mem_size_output);

    // --- sum_cluster
    unsigned int size_sum = K*3;
    unsigned int mem_size_sum = sizeof(float) * size_sum;
    float* h_sum = (float*) malloc(mem_size_sum);

    // -- count_cluster
    unsigned int size_count = K;
    unsigned int mem_size_count = sizeof(int) * size_count;
    int* h_count = (int*) malloc(mem_size_count);

    // --- old/new cluster
    unsigned int size_centroid = K*3;
    unsigned int mem_size_centroid = sizeof(float) * size_centroid;
    float* h_old = (float*) malloc(mem_size_centroid);
    float* h_new = (float*) malloc(mem_size_centroid);


    // Device Allocate
    // --- K
    // --- Height & Width

    // --- Image
    float* d_img;
    cudaMalloc((void***) &d_img, mem_size_img);

    // --- output_cluster
    int* d_output;
    cudaMalloc((void**) &d_output, mem_size_output);
    
    // --- sum_cluster
    // float* d_sum;
    // cudaMalloc((void**) &d_sum, mem_size_sum);
    
    // --- count_cluster
    // float* d_count;
    // cudaMalloc(&d_count, mem_size_count);
    
    // --- old_cluster
    float* d_old;
    cudaMalloc((void**) &d_old, mem_size_centroid);

    // Inintial "h_img"
    for(int i=0 ; i<height ; i++){
        for(int j=0 ; j<width ; j++){
            for(int k=0 ; k<3 ; k++){
                h_img[i*width + j*3 + k] = 5;
            }
        }
    }

    srand(time(0));
    // Inintial "h_sum" & "h_count" & "h_old"
    for(int i=0 ; i<K ; i++){
        for(int j=0 ; j<3 ; j++){
            h_sum[i*3 + j] = 0;
            h_old[i*3 + j] = rand() % 256;
            // h_old[i*3 + j] = 4; 
        }
        h_count[i] = 0;
    }

    // printf("[%f]\n", h_old[0*3 + 0]);
    // printf("[%f]\n", h_img[0*3 + 0]);

    // Copy Memory from host to Device
    cudaMemcpy(d_img, h_img, mem_size_img, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, mem_size_output, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_sum, h_sum, mem_size_sum, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_count, h_count, mem_size_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old, h_old, mem_size_centroid, cudaMemcpyHostToDevice);

    // Set Grid & Block Size
    dim3 blockSize, gridSize;
    blockSize = dim3(64,4);
    gridSize = dim3(height/(64*4) , width/(64));

    // Computation
    hsv_k_mean<<<gridSize, blockSize>>>(d_output, K, d_old,
                                        d_img, height, width);

    cudaStreamSynchronize(0);
           
    cudaMemcpy(h_output, d_output, mem_size_output, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_sum, d_sum, mem_size_sum, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_count, d_count, mem_size_count, cudaMemcpyDeviceToHost);

    // Sum and count
    for(int i=0 ; i<height ; i++){
        for(int j=0 ; j<width ; j++){
            // printf("%d\n", h_output[i*width + j]);

            h_sum[h_output[i*width + j]*K + 0] += h_img[i*width + j*3 + 0];   // Hue
            h_sum[h_output[i*width + j]*K + 1] += h_img[i*width + j*3 + 1];   // Satuation
            h_sum[h_output[i*width + j]*K + 2] += h_img[i*width + j*3 + 2];   // Value
            h_count[h_output[i*width + j]] += 1;

            // printf("%d\n", h_count[h_output[i*width + j]]);
        }
    }

    // Calculate New Centroids
    for(int i=0 ; i<K ; i++){
        if(h_count[i] == 0){
            h_new[i*3 + 0] = 0;                           // Hue
            h_new[i*3 + 1] = 0;                           // Satuation
            h_new[i*3 + 2] = 0;                            // Value
        } else{
            h_new[i*3 + 0] = h_sum[i*3 + 0]/h_count[i];   // Hue
            h_new[i*3 + 1] = h_sum[i*3 + 1]/h_count[i];   // Satuation
            h_new[i*3 + 2] = h_sum[i*3 + 2]/h_count[i];   // Value
        }

        // printf("%d\n", h_count[i]);
        printf("[%f, %f, %f]\n", h_new[i*3 + 0], h_new[i*3 + 1], h_new[i*3 + 2]);
    }

    // Free All data
    free(h_img);      cudaFree(d_img);
    free(h_output);   cudaFree(d_output);
    free(h_sum);      //cudaFree(d_sum);
    free(h_count);    //cudaFree(d_count);
    free(h_old);      cudaFree(d_old);

    return 0;
}
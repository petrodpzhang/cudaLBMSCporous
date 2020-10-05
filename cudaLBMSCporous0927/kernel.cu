#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <algorithm>

using namespace std;

// 1. constant varibles

const int Nx = 96;
const int Ny = 96;
const int Nz = 96;
int Nlattice = Nx * Ny * Nz;
const double tauA = 1.0;
const double tauB = 1.0;

// 2. GPU and CPU arrays

int* d_geo, * h_geo;
double* d_ux_1, * h_ux_1, * d_ux_2, * h_ux_2;
double* d_uy_1, * h_uy_1, * d_uy_2, * h_uy_2;
double* d_uz_1, * h_uz_1, * d_uz_2, * h_uz_2;
double* d_rho_1, * h_rho_1;
double* d_rho_2, * h_rho_2;
double* d_psi_1, * h_psi_1;
double* d_psi_2, * h_psi_2;
double* d_f_1, * h_f_1;
double* d_f_2, * h_f_2;
double* d_f_post_1, * h_f_post_1;
double* d_f_post_2, * h_f_post_2;
double* d_Fx_1;
double* d_Fx_2;
double* d_Fy_1;
double* d_Fy_2;
double* d_Fz_1;
double* d_Fz_2;
double* press, * testq, * testd;
int* test;

// 3.D3Q19

const int Q = 19;
const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };

// 4. Assign a 3D distribution of CUDA

int threadsAlongX = 8;
int threadsAlongY = 8;
int threadsAlongZ = 8;

dim3 block(threadsAlongX, threadsAlongY, threadsAlongZ);
dim3 grid(1 + (Nx - 1) / threadsAlongX, 1 + (Ny - 1) / threadsAlongY, 1 + (Nz - 1) / threadsAlongZ);

// 5. prepare geometry

void pre_geo()
{
    int LXC[8], LYC[8], LZC[8];
    int R2 = 22;
    int indexf[19];

    for (int i = 0; i < 4; i++)
    {
        LZC[i] = 23;
    }
    LXC[0] = 23;
    LYC[0] = 23;
    LXC[1] = 23 + 2 * R2;
    LYC[1] = 23;
    LXC[2] = 23;
    LYC[2] = 23 + 2 * R2;
    LXC[3] = 23 + 2 * R2;
    LYC[3] = 23 + 2 * R2;

    for (int i = 4; i < 8; i++)
    {
        LZC[i] = 23 + 2 * R2;
    }
    LXC[4] = 23;
    LYC[4] = 23;
    LXC[5] = 23 + 2 * R2;
    LYC[5] = 23;
    LXC[6] = 23;
    LYC[6] = 23 + 2 * R2;
    LXC[7] = 23 + 2 * R2;
    LYC[7] = 23 + 2 * R2;

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                int index = z * Nx * Ny + y * Nx + x;

                h_geo[index] = 0;

                for (int L = 0; L < 8; L++)
                {
                    if ((x - LXC[L]) * (x - LXC[L]) + (y - LYC[L]) * (y - LYC[L]) + (z - LZC[L]) * (z - LZC[L]) < R2 * R2)
                        h_geo[index] = 1;
                }
            }
        }
    }

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                int index = z * Nx * Ny + y * Nx + x;
                if (h_geo[index] == 0)
                    for (int q = 0; q < 19; q++)
                    {
                        int i_1 = (x + cx[q] + Nx) % Nx;
                        int j_1 = (y + cy[q] + Ny) % Ny;
                        int k_1 = (z + cz[q] + Nz) % Nz;
                        indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;
                        if (h_geo[indexf[q]] == 1) // boundary
                            h_geo[index] = 2;
                        else if (z == 0) // inlet
                            h_geo[index] = 3;
                        else if (z == Nx - 1) // outlet
                            h_geo[index] = 4;
                    }
            }
        }
    }
}

// 6. initialize

void Initialization(int* h_geo)
{
    double feq_1[19], feq_2[19];
    double tmp_rho_1, tmp_rho_2;
    double tmp_ux_1, tmp_uy_1, tmp_uz_1, ux2_1, uy2_1, uz2_1, uxyz2_1, uxy2_1, uxz2_1, uyz2_1, uxy_1, uxz_1, uyz_1;
    double tmp_ux_2, tmp_uy_2, tmp_uz_2, ux2_2, uy2_2, uz2_2, uxyz2_2, uxy2_2, uxz2_2, uyz2_2, uxy_2, uxz_2, uyz_2;


    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                int index = z * Nx * Ny + y * Nx + x;

                h_ux_1[index] = 0.0;
                h_uy_1[index] = 0.0;
                h_uz_1[index] = 0.0;
                h_ux_2[index] = 0.0;
                h_uy_2[index] = 0.0;
                h_uz_2[index] = 0.0;
                h_psi_1[index] = 0.0;
                h_psi_2[index] = 0.0;

                if (h_geo[index] == 0 || h_geo[index] == 2 || h_geo[index] == 3 || h_geo[index] == 4)
                {
                    if (z < 2)
                    {
                        h_rho_1[index] = 1.6;
                        h_rho_2[index] = 0.006;
                    }
                    else
                    {
                        h_rho_1[index] = 0.006;
                        h_rho_2[index] = 8.0;
                    }
                }
                else
                    if (h_geo[index] == 1)
                    {
                        h_rho_1[index] = 0.0;
                        h_rho_2[index] = 0.0;
                    }
            }
        }
    }

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                int index = z * Nx * Ny + y * Nx + x;

                tmp_rho_1 = h_rho_1[index];
                tmp_rho_2 = h_rho_2[index];

                tmp_ux_1 = h_ux_1[index];
                tmp_uy_1 = h_uy_1[index];
                tmp_uz_1 = h_uz_1[index];
                ux2_1 = tmp_ux_1 * tmp_ux_1;
                uy2_1 = tmp_uy_1 * tmp_uy_1;
                uz2_1 = tmp_uz_1 * tmp_uz_1;
                uxyz2_1 = ux2_1 + uy2_1 + uz2_1;
                uxy2_1 = ux2_1 + uy2_1;
                uxz2_1 = ux2_1 + uz2_1;
                uyz2_1 = uy2_1 + uz2_1;
                uxy_1 = 2.0 * tmp_ux_1 * tmp_uy_1;
                uxz_1 = 2.0 * tmp_ux_1 * tmp_uz_1;
                uyz_1 = 2.0 * tmp_uy_1 * tmp_uz_1;

                tmp_ux_2 = h_ux_2[index];
                tmp_uy_2 = h_uy_2[index];
                tmp_uz_2 = h_uz_2[index];
                ux2_2 = tmp_ux_2 * tmp_ux_2;
                uy2_2 = tmp_uy_2 * tmp_uy_2;
                uz2_2 = tmp_uz_2 * tmp_uz_2;
                uxyz2_2 = ux2_2 + uy2_2 + uz2_2;
                uxy2_2 = ux2_2 + uy2_2;
                uxz2_2 = ux2_2 + uz2_2;
                uyz2_2 = uy2_2 + uz2_2;
                uxy_2 = 2.0 * tmp_ux_2 * tmp_uy_2;
                uxz_2 = 2.0 * tmp_ux_2 * tmp_uz_2;
                uyz_2 = 2.0 * tmp_uy_2 * tmp_uz_2;

                feq_1[0] = tmp_rho_1 * w[0] * (1.0 - 1.5 * uxyz2_1);
                feq_1[1] = tmp_rho_1 * w[1] * (1.0 + 3.0 * tmp_ux_1 + 4.5 * ux2_1 - 1.5 * uxyz2_1);
                feq_1[2] = tmp_rho_1 * w[2] * (1.0 - 3.0 * tmp_ux_1 + 4.5 * ux2_1 - 1.5 * uxyz2_1);
                feq_1[3] = tmp_rho_1 * w[3] * (1.0 + 3.0 * tmp_uy_1 + 4.5 * uy2_1 - 1.5 * uxyz2_1);
                feq_1[4] = tmp_rho_1 * w[4] * (1.0 - 3.0 * tmp_uy_1 + 4.5 * uy2_1 - 1.5 * uxyz2_1);
                feq_1[5] = tmp_rho_1 * w[5] * (1.0 + 3.0 * tmp_uz_1 + 4.5 * uz2_1 - 1.5 * uxyz2_1);
                feq_1[6] = tmp_rho_1 * w[6] * (1.0 - 3.0 * tmp_uz_1 + 4.5 * uz2_1 - 1.5 * uxyz2_1);
                feq_1[7] = tmp_rho_1 * w[7] * (1.0 + 3.0 * (tmp_ux_1 + tmp_uy_1) + 4.5 * (uxy2_1 + uxy_1) - 1.5 * uxyz2_1);
                feq_1[8] = tmp_rho_1 * w[8] * (1.0 + 3.0 * (tmp_ux_1 - tmp_uy_1) + 4.5 * (uxy2_1 - uxy_1) - 1.5 * uxyz2_1);
                feq_1[9] = tmp_rho_1 * w[9] * (1.0 + 3.0 * (tmp_uy_1 - tmp_ux_1) + 4.5 * (uxy2_1 - uxy_1) - 1.5 * uxyz2_1);
                feq_1[10] = tmp_rho_1 * w[10] * (1.0 - 3.0 * (tmp_ux_1 + tmp_uy_1) + 4.5 * (uxy2_1 + uxy_1) - 1.5 * uxyz2_1);
                feq_1[11] = tmp_rho_1 * w[11] * (1.0 + 3.0 * (tmp_ux_1 + tmp_uz_1) + 4.5 * (uxz2_1 + uxz_1) - 1.5 * uxyz2_1);
                feq_1[12] = tmp_rho_1 * w[12] * (1.0 + 3.0 * (tmp_ux_1 - tmp_uz_1) + 4.5 * (uxz2_1 - uxz_1) - 1.5 * uxyz2_1);
                feq_1[13] = tmp_rho_1 * w[13] * (1.0 + 3.0 * (tmp_uz_1 - tmp_ux_1) + 4.5 * (uxz2_1 - uxz_1) - 1.5 * uxyz2_1);
                feq_1[14] = tmp_rho_1 * w[14] * (1.0 - 3.0 * (tmp_ux_1 + tmp_uz_1) + 4.5 * (uxz2_1 + uxz_1) - 1.5 * uxyz2_1);
                feq_1[15] = tmp_rho_1 * w[15] * (1.0 + 3.0 * (tmp_uy_1 + tmp_uz_1) + 4.5 * (uyz2_1 + uyz_1) - 1.5 * uxyz2_1);
                feq_1[16] = tmp_rho_1 * w[16] * (1.0 + 3.0 * (tmp_uz_1 - tmp_uy_1) + 4.5 * (uyz2_1 - uyz_1) - 1.5 * uxyz2_1);
                feq_1[17] = tmp_rho_1 * w[17] * (1.0 + 3.0 * (tmp_uy_1 - tmp_uz_1) + 4.5 * (uyz2_1 - uyz_1) - 1.5 * uxyz2_1);
                feq_1[18] = tmp_rho_1 * w[18] * (1.0 - 3.0 * (tmp_uy_1 + tmp_uz_1) + 4.5 * (uyz2_1 + uyz_1) - 1.5 * uxyz2_1);

                feq_2[0] = tmp_rho_2 * w[0] * (1.0 - 1.5 * uxyz2_2);
                feq_2[1] = tmp_rho_2 * w[1] * (1.0 + 3.0 * tmp_ux_2 + 4.5 * ux2_2 - 1.5 * uxyz2_2);
                feq_2[2] = tmp_rho_2 * w[2] * (1.0 - 3.0 * tmp_ux_2 + 4.5 * ux2_2 - 1.5 * uxyz2_2);
                feq_2[3] = tmp_rho_2 * w[3] * (1.0 + 3.0 * tmp_uy_2 + 4.5 * uy2_2 - 1.5 * uxyz2_2);
                feq_2[4] = tmp_rho_2 * w[4] * (1.0 - 3.0 * tmp_uy_2 + 4.5 * uy2_2 - 1.5 * uxyz2_2);
                feq_2[5] = tmp_rho_2 * w[5] * (1.0 + 3.0 * tmp_uz_2 + 4.5 * uz2_2 - 1.5 * uxyz2_2);
                feq_2[6] = tmp_rho_2 * w[6] * (1.0 - 3.0 * tmp_uz_2 + 4.5 * uz2_2 - 1.5 * uxyz2_2);
                feq_2[7] = tmp_rho_2 * w[7] * (1.0 + 3.0 * (tmp_ux_2 + tmp_uy_2) + 4.5 * (uxy2_2 + uxy_2) - 1.5 * uxyz2_2);
                feq_2[8] = tmp_rho_2 * w[8] * (1.0 + 3.0 * (tmp_ux_2 - tmp_uy_2) + 4.5 * (uxy2_2 - uxy_2) - 1.5 * uxyz2_2);
                feq_2[9] = tmp_rho_2 * w[9] * (1.0 + 3.0 * (tmp_uy_2 - tmp_ux_2) + 4.5 * (uxy2_2 - uxy_2) - 1.5 * uxyz2_2);
                feq_2[10] = tmp_rho_2 * w[10] * (1.0 - 3.0 * (tmp_ux_2 + tmp_uy_2) + 4.5 * (uxy2_2 + uxy_2) - 1.5 * uxyz2_2);
                feq_2[11] = tmp_rho_2 * w[11] * (1.0 + 3.0 * (tmp_ux_2 + tmp_uz_2) + 4.5 * (uxz2_2 + uxz_2) - 1.5 * uxyz2_2);
                feq_2[12] = tmp_rho_2 * w[12] * (1.0 + 3.0 * (tmp_ux_2 - tmp_uz_2) + 4.5 * (uxz2_2 - uxz_2) - 1.5 * uxyz2_2);
                feq_2[13] = tmp_rho_2 * w[13] * (1.0 + 3.0 * (tmp_uz_2 - tmp_ux_2) + 4.5 * (uxz2_2 - uxz_2) - 1.5 * uxyz2_2);
                feq_2[14] = tmp_rho_2 * w[14] * (1.0 - 3.0 * (tmp_ux_2 + tmp_uz_2) + 4.5 * (uxz2_2 + uxz_2) - 1.5 * uxyz2_2);
                feq_2[15] = tmp_rho_2 * w[15] * (1.0 + 3.0 * (tmp_uy_2 + tmp_uz_2) + 4.5 * (uyz2_2 + uyz_2) - 1.5 * uxyz2_2);
                feq_2[16] = tmp_rho_2 * w[16] * (1.0 + 3.0 * (tmp_uz_2 - tmp_uy_2) + 4.5 * (uyz2_2 - uyz_2) - 1.5 * uxyz2_2);
                feq_2[17] = tmp_rho_2 * w[17] * (1.0 + 3.0 * (tmp_uy_2 - tmp_uz_2) + 4.5 * (uyz2_2 - uyz_2) - 1.5 * uxyz2_2);
                feq_2[18] = tmp_rho_2 * w[18] * (1.0 - 3.0 * (tmp_uy_2 + tmp_uz_2) + 4.5 * (uyz2_2 + uyz_2) - 1.5 * uxyz2_2);

                if (h_geo[index] == 0 || h_geo[index] == 2 || h_geo[index] == 3 || h_geo[index] == 4)
                {
                    for (int q = 0; q < Q; q++)
                    {
                        h_f_1[Nlattice * q + index] = feq_1[q];
                        h_f_post_1[Nlattice * q + index] = feq_1[q];
                        h_f_2[Nlattice * q + index] = feq_2[q];
                        h_f_post_2[Nlattice * q + index] = feq_2[q];
                    }
                }
                else
                    if (h_geo[index] == 1)
                    {
                        for (int q = 0; q < Q; q++)
                        {
                            h_f_1[Nlattice * q + index] = 0.0;
                            h_f_post_1[Nlattice * q + index] = 0.0;
                            h_f_2[Nlattice * q + index] = 0.0;
                            h_f_post_2[Nlattice * q + index] = 0.0;
                        }
                    }
            }
        }
    }
}

// 7. Density

__global__ void computeDensity(int Nlattice, int* __restrict__ d_geo, double* __restrict__ d_f_1, double* __restrict__ d_f_2, double* __restrict__ d_rho_1, double* __restrict__ d_rho_2, double* __restrict__ d_psi_1, double* __restrict__ d_psi_2)
{
    const int Q = 19;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;

    if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4)
    {
        for (int q = 0; q < Q; q++)
        {
            d_rho_1[index] += d_f_1[index + Nlattice * q];
            d_rho_2[index] += d_f_2[index + Nlattice * q];
        }
        
        d_psi_1[index] = d_rho_1[index];
        d_psi_2[index] = d_rho_2[index];
        //const int rho0 = 1.0;
        //d_psi_1[index] = rho0 * (1.0 - exp(-d_rho_1[index] / rho0));
        //d_psi_2[index] = rho0 * (1.0 - exp(-d_rho_2[index] / rho0));
    }
    else if (d_geo[index] == 1)
    {
        d_rho_1[index] = 0.0;
        d_rho_2[index] = 0.0;
        d_psi_1[index] = 0.0;
        d_psi_2[index] = 0.0;
    }       
}

// 8. Force

__global__ void computeSCForces(int* __restrict__ d_geo, double* __restrict__ d_psi_1, double* __restrict__ d_psi_2, double* __restrict__ d_Fx_1, double* __restrict__ d_Fy_1, double* __restrict__ d_Fz_1, double* __restrict__ d_Fx_2, double* __restrict__ d_Fy_2, double* __restrict__ d_Fz_2)
{
    const double rho_in = 8.2;
    const double rho_out = 7.8;
    const double rho_s = 0.006;
    const double Gc = 0.225;//0.9 0.225
    const double Gs1 = 0.9;
    const double Gs2 = -0.9;
    int indexf[19];

    const int Q = 19;
    const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
    const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
    const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
    const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;

    d_Fx_1[index] = 0.0;
    d_Fx_2[index] = 0.0;
    d_Fy_1[index] = 0.0;
    d_Fy_2[index] = 0.0;
    d_Fz_1[index] = 0.0;
    d_Fz_2[index] = 0.0;

    if (d_geo[index] == 0) // fluid-fluid force
        for (int q = 0; q < 19; q++)
        {
            int i_1 = (i + cx[q] + Nx) % Nx;
            int j_1 = (j + cy[q] + Ny) % Ny;
            int k_1 = (k + cz[q] + Nz) % Nz;
            indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;

            d_Fx_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cx[q];
            d_Fx_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cx[q];
            d_Fy_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cy[q];
            d_Fy_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cy[q];
            d_Fz_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cz[q];
            d_Fz_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cz[q];
        }
    else if (d_geo[index] == 2) // fluid-solid force
        for (int q = 0; q < 19; q++)
        {
            int i_1 = (i + cx[q] + Nx) % Nx;
            int j_1 = (j + cy[q] + Ny) % Ny;
            int k_1 = (k + cz[q] + Nz) % Nz;
            indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;

            d_Fx_1[index] += (1.0 - d_geo[indexf[q]]) * Gc * (-1.0) * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cx[q] + d_geo[indexf[q]] * Gs1 * (-1.0) * d_psi_1[index] * w[q] * cx[q];
            d_Fx_2[index] += (1.0 - d_geo[indexf[q]]) * Gc * (-1.0) * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cx[q] + d_geo[indexf[q]] * Gs2 * (-1.0) * d_psi_2[index] * w[q] * cx[q];
            d_Fy_1[index] += (1.0 - d_geo[indexf[q]]) * Gc * (-1.0) * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cy[q] + d_geo[indexf[q]] * Gs1 * (-1.0) * d_psi_1[index] * w[q] * cy[q];
            d_Fy_2[index] += (1.0 - d_geo[indexf[q]]) * Gc * (-1.0) * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cy[q] + d_geo[indexf[q]] * Gs2 * (-1.0) * d_psi_2[index] * w[q] * cy[q];
            d_Fz_1[index] += (1.0 - d_geo[indexf[q]]) * Gc * (-1.0) * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cz[q] + d_geo[indexf[q]] * Gs1 * (-1.0) * d_psi_1[index] * w[q] * cz[q];
            d_Fz_2[index] += (1.0 - d_geo[indexf[q]]) * Gc * (-1.0) * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cz[q] + d_geo[indexf[q]] * Gs2 * (-1.0) * d_psi_2[index] * w[q] * cz[q];
        }
    else if (d_geo[index] == 3)//inlet
    {
        d_Fz_1[index] = (-1.0) * Gc * d_psi_1[index] * (1.0 / 18 + 4.0 / 36) * rho_s * (-1.0);
        d_Fz_2[index] = (-1.0) * Gc * d_psi_2[index] * (1.0 / 18 + 4.0 / 36) * rho_in * (-1.0);

        for (int q = 0; q < 19; q++)
        {
            int i_1 = (i + cx[q] + Nx) % Nx;
            int j_1 = (j + cy[q] + Ny) % Ny;
            int k_1 = (k + cz[q] + Nz) % Nz;
            indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;
            if (q == 6 || q == 10 || q == 12 || q == 15 || q == 17)
                continue;
            d_Fx_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cx[q];
            d_Fx_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cx[q];
            d_Fy_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cy[q];
            d_Fy_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cy[q];
            //d_Fz_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cz[q];
            //d_Fz_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cz[q];
        }
    }
    else if (d_geo[index] == 4)//outlet
    {
        d_Fz_1[index] = (-1.0) * Gc * d_psi_1[index] * (1.0 / 18 + 4.0 / 36) * rho_out;
        d_Fz_2[index] = (-1.0) * Gc * d_psi_2[index] * (1.0 / 18 + 4.0 / 36) * rho_s;

        for (int q = 0; q < 19; q++)
        {
            int i_1 = (i + cx[q] + Nx) % Nx;
            int j_1 = (j + cy[q] + Ny) % Ny;
            int k_1 = (k + cz[q] + Nz) % Nz;
            indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;
            if (q == 5 || q == 9 || q == 11 || q == 16 || q == 18)
                continue;
            d_Fx_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cx[q];
            d_Fx_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cx[q];
            d_Fy_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cy[q];
            d_Fy_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cy[q];
            //d_Fz_1[index] += (-1.0) * Gc * d_psi_1[index] * w[q] * d_psi_2[indexf[q]] * cz[q];
            //d_Fz_2[index] += (-1.0) * Gc * d_psi_2[index] * w[q] * d_psi_1[indexf[q]] * cz[q];
        }
    }       
}

// 9. velocity

__global__ void computeVelocity(int* __restrict__ d_geo, double* __restrict__ d_f_1, double* __restrict__ d_f_2, double* __restrict__ d_rho_1, double* __restrict__ d_rho_2, 
    double* __restrict__ d_Fx_1, double* __restrict__ d_Fy_1, double* __restrict__ d_Fz_1, double* __restrict__ d_Fx_2, double* __restrict__ d_Fy_2, double* __restrict__ d_Fz_2, 
    double* __restrict__ d_ux_1, double* __restrict__ d_uy_1, double* __restrict__ d_uz_1, double* __restrict__ d_ux_2, double* __restrict__ d_uy_2, double* __restrict__ d_uz_2)
{
    const double tauA = 1.0;
    const double tauB = 1.0;
    const int Q = 19;
    const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
    const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
    const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
    const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;
    int Nlattice = Nx * Ny * Nz;

    if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4)
    {
        d_ux_1[index] = tauA * d_Fx_1[index] / d_rho_1[index];
        d_ux_2[index] = tauB * d_Fx_2[index] / d_rho_2[index];
        d_uy_1[index] = tauA * d_Fy_1[index] / d_rho_1[index];
        d_uy_2[index] = tauB * d_Fy_2[index] / d_rho_2[index];
        d_uz_1[index] = tauA * d_Fz_1[index] / d_rho_1[index];
        d_uz_2[index] = tauB * d_Fz_2[index] / d_rho_2[index];
        for (int q = 0; q < 19; q++)
        {
            d_ux_1[index] += (d_f_1[index + Nlattice * q] / tauA + d_f_2[index + Nlattice * q] / tauB) * cx[q] / (d_rho_1[index] / tauA + d_rho_2[index] / tauB);
            d_ux_2[index] += (d_f_1[index + Nlattice * q] / tauA + d_f_2[index + Nlattice * q] / tauB) * cx[q] / (d_rho_1[index] / tauA + d_rho_2[index] / tauB);
            d_uy_1[index] += (d_f_1[index + Nlattice * q] / tauA + d_f_2[index + Nlattice * q] / tauB) * cy[q] / (d_rho_1[index] / tauA + d_rho_2[index] / tauB);
            d_uy_2[index] += (d_f_1[index + Nlattice * q] / tauA + d_f_2[index + Nlattice * q] / tauB) * cy[q] / (d_rho_1[index] / tauA + d_rho_2[index] / tauB);
            d_uz_1[index] += (d_f_1[index + Nlattice * q] / tauA + d_f_2[index + Nlattice * q] / tauB) * cz[q] / (d_rho_1[index] / tauA + d_rho_2[index] / tauB);
            d_uz_2[index] += (d_f_1[index + Nlattice * q] / tauA + d_f_2[index + Nlattice * q] / tauB) * cz[q] / (d_rho_1[index] / tauA + d_rho_2[index] / tauB);
        }        
    }
}

// 10. stream

__global__ void StreamBoundary(int* __restrict__ d_geo, double* __restrict__ d_f_1, double* __restrict__ d_f_2, double* __restrict__ d_f_post_1, double* __restrict__ d_f_post_2, 
    double* __restrict__ d_Fx_1, double* __restrict__ d_Fy_1, double* __restrict__ d_Fz_1, double* __restrict__ d_Fx_2, double* __restrict__ d_Fy_2, double* __restrict__ d_Fz_2, 
    double* __restrict__ d_ux_1, double* __restrict__ d_uy_1, double* __restrict__ d_uz_1, double* __restrict__ d_ux_2, double* __restrict__ d_uy_2, double* __restrict__ d_uz_2, 
    double* __restrict__ d_rho_1, double* __restrict__ d_rho_2)
{
    const double rho_in = 8.2;
    const double rho_out = 7.8;
    const double rho_s = 0.006;
    const double tauA = 1.0;
    const double tauB = 1.0;
    
    double feq_1[19], feq_2[19];
    double tmp_rho_1, tmp_rho_2;
    double tmp_ux_1, tmp_uy_1, tmp_uz_1, ux2_1, uy2_1, uz2_1, uxyz2_1, uxy2_1, uxz2_1, uyz2_1, uxy_1, uxz_1, uyz_1;
    double tmp_ux_2, tmp_uy_2, tmp_uz_2, ux2_2, uy2_2, uz2_2, uxyz2_2, uxy2_2, uxz2_2, uyz2_2, uxy_2, uxz_2, uyz_2;
    int indexf[19];
    const int Q = 19;
    const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
    const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
    const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
    const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };


    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;
    int Nlattice = Nx * Ny * Nz;

    for (int q = 0; q < 19; q++)
    {
        int i_1 = (i + cx[q] + Nx) % Nx;
        int j_1 = (j + cy[q] + Ny) % Ny;
        int k_1 = (k + cz[q] + Nz) % Nz;
        indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;
    }

    tmp_rho_1 = d_rho_1[index];
    tmp_rho_2 = d_rho_2[index];

    tmp_ux_1 = d_ux_1[index];
    tmp_uy_1 = d_uy_1[index];
    tmp_uz_1 = d_uz_1[index];
    ux2_1 = tmp_ux_1 * tmp_ux_1;
    uy2_1 = tmp_uy_1 * tmp_uy_1;
    uz2_1 = tmp_uz_1 * tmp_uz_1;
    uxyz2_1 = ux2_1 + uy2_1 + uz2_1;
    uxy2_1 = ux2_1 + uy2_1;
    uxz2_1 = ux2_1 + uz2_1;
    uyz2_1 = uy2_1 + uz2_1;
    uxy_1 = 2.0 * tmp_ux_1 * tmp_uy_1;
    uxz_1 = 2.0 * tmp_ux_1 * tmp_uz_1;
    uyz_1 = 2.0 * tmp_uy_1 * tmp_uz_1;

    tmp_ux_2 = d_ux_2[index];
    tmp_uy_2 = d_uy_2[index];
    tmp_uz_2 = d_uz_2[index];
    ux2_2 = tmp_ux_2 * tmp_ux_2;
    uy2_2 = tmp_uy_2 * tmp_uy_2;
    uz2_2 = tmp_uz_2 * tmp_uz_2;
    uxyz2_2 = ux2_2 + uy2_2 + uz2_2;
    uxy2_2 = ux2_2 + uy2_2;
    uxz2_2 = ux2_2 + uz2_2;
    uyz2_2 = uy2_2 + uz2_2;
    uxy_2 = 2.0 * tmp_ux_2 * tmp_uy_2;
    uxz_2 = 2.0 * tmp_ux_2 * tmp_uz_2;
    uyz_2 = 2.0 * tmp_uy_2 * tmp_uz_2;

    feq_1[0] = tmp_rho_1 * w[0] * (1.0 - 1.5 * uxyz2_1);
    feq_1[1] = tmp_rho_1 * w[1] * (1.0 + 3.0 * tmp_ux_1 + 4.5 * ux2_1 - 1.5 * uxyz2_1);
    feq_1[2] = tmp_rho_1 * w[2] * (1.0 - 3.0 * tmp_ux_1 + 4.5 * ux2_1 - 1.5 * uxyz2_1);
    feq_1[3] = tmp_rho_1 * w[3] * (1.0 + 3.0 * tmp_uy_1 + 4.5 * uy2_1 - 1.5 * uxyz2_1);
    feq_1[4] = tmp_rho_1 * w[4] * (1.0 - 3.0 * tmp_uy_1 + 4.5 * uy2_1 - 1.5 * uxyz2_1);
    feq_1[5] = tmp_rho_1 * w[5] * (1.0 + 3.0 * tmp_uz_1 + 4.5 * uz2_1 - 1.5 * uxyz2_1);
    feq_1[6] = tmp_rho_1 * w[6] * (1.0 - 3.0 * tmp_uz_1 + 4.5 * uz2_1 - 1.5 * uxyz2_1);
    feq_1[7] = tmp_rho_1 * w[7] * (1.0 + 3.0 * (tmp_ux_1 + tmp_uy_1) + 4.5 * (uxy2_1 + uxy_1) - 1.5 * uxyz2_1);
    feq_1[8] = tmp_rho_1 * w[8] * (1.0 + 3.0 * (tmp_ux_1 - tmp_uy_1) + 4.5 * (uxy2_1 - uxy_1) - 1.5 * uxyz2_1);
    feq_1[9] = tmp_rho_1 * w[9] * (1.0 + 3.0 * (tmp_uy_1 - tmp_ux_1) + 4.5 * (uxy2_1 - uxy_1) - 1.5 * uxyz2_1);
    feq_1[10] = tmp_rho_1 * w[10] * (1.0 - 3.0 * (tmp_ux_1 + tmp_uy_1) + 4.5 * (uxy2_1 + uxy_1) - 1.5 * uxyz2_1);
    feq_1[11] = tmp_rho_1 * w[11] * (1.0 + 3.0 * (tmp_ux_1 + tmp_uz_1) + 4.5 * (uxz2_1 + uxz_1) - 1.5 * uxyz2_1);
    feq_1[12] = tmp_rho_1 * w[12] * (1.0 + 3.0 * (tmp_ux_1 - tmp_uz_1) + 4.5 * (uxz2_1 - uxz_1) - 1.5 * uxyz2_1);
    feq_1[13] = tmp_rho_1 * w[13] * (1.0 + 3.0 * (tmp_uz_1 - tmp_ux_1) + 4.5 * (uxz2_1 - uxz_1) - 1.5 * uxyz2_1);
    feq_1[14] = tmp_rho_1 * w[14] * (1.0 - 3.0 * (tmp_ux_1 + tmp_uz_1) + 4.5 * (uxz2_1 + uxz_1) - 1.5 * uxyz2_1);
    feq_1[15] = tmp_rho_1 * w[15] * (1.0 + 3.0 * (tmp_uy_1 + tmp_uz_1) + 4.5 * (uyz2_1 + uyz_1) - 1.5 * uxyz2_1);
    feq_1[16] = tmp_rho_1 * w[16] * (1.0 + 3.0 * (tmp_uz_1 - tmp_uy_1) + 4.5 * (uyz2_1 - uyz_1) - 1.5 * uxyz2_1);
    feq_1[17] = tmp_rho_1 * w[17] * (1.0 + 3.0 * (tmp_uy_1 - tmp_uz_1) + 4.5 * (uyz2_1 - uyz_1) - 1.5 * uxyz2_1);
    feq_1[18] = tmp_rho_1 * w[18] * (1.0 - 3.0 * (tmp_uy_1 + tmp_uz_1) + 4.5 * (uyz2_1 + uyz_1) - 1.5 * uxyz2_1);

    feq_2[0] = tmp_rho_2 * w[0] * (1.0 - 1.5 * uxyz2_2);
    feq_2[1] = tmp_rho_2 * w[1] * (1.0 + 3.0 * tmp_ux_2 + 4.5 * ux2_2 - 1.5 * uxyz2_2);
    feq_2[2] = tmp_rho_2 * w[2] * (1.0 - 3.0 * tmp_ux_2 + 4.5 * ux2_2 - 1.5 * uxyz2_2);
    feq_2[3] = tmp_rho_2 * w[3] * (1.0 + 3.0 * tmp_uy_2 + 4.5 * uy2_2 - 1.5 * uxyz2_2);
    feq_2[4] = tmp_rho_2 * w[4] * (1.0 - 3.0 * tmp_uy_2 + 4.5 * uy2_2 - 1.5 * uxyz2_2);
    feq_2[5] = tmp_rho_2 * w[5] * (1.0 + 3.0 * tmp_uz_2 + 4.5 * uz2_2 - 1.5 * uxyz2_2);
    feq_2[6] = tmp_rho_2 * w[6] * (1.0 - 3.0 * tmp_uz_2 + 4.5 * uz2_2 - 1.5 * uxyz2_2);
    feq_2[7] = tmp_rho_2 * w[7] * (1.0 + 3.0 * (tmp_ux_2 + tmp_uy_2) + 4.5 * (uxy2_2 + uxy_2) - 1.5 * uxyz2_2);
    feq_2[8] = tmp_rho_2 * w[8] * (1.0 + 3.0 * (tmp_ux_2 - tmp_uy_2) + 4.5 * (uxy2_2 - uxy_2) - 1.5 * uxyz2_2);
    feq_2[9] = tmp_rho_2 * w[9] * (1.0 + 3.0 * (tmp_uy_2 - tmp_ux_2) + 4.5 * (uxy2_2 - uxy_2) - 1.5 * uxyz2_2);
    feq_2[10] = tmp_rho_2 * w[10] * (1.0 - 3.0 * (tmp_ux_2 + tmp_uy_2) + 4.5 * (uxy2_2 + uxy_2) - 1.5 * uxyz2_2);
    feq_2[11] = tmp_rho_2 * w[11] * (1.0 + 3.0 * (tmp_ux_2 + tmp_uz_2) + 4.5 * (uxz2_2 + uxz_2) - 1.5 * uxyz2_2);
    feq_2[12] = tmp_rho_2 * w[12] * (1.0 + 3.0 * (tmp_ux_2 - tmp_uz_2) + 4.5 * (uxz2_2 - uxz_2) - 1.5 * uxyz2_2);
    feq_2[13] = tmp_rho_2 * w[13] * (1.0 + 3.0 * (tmp_uz_2 - tmp_ux_2) + 4.5 * (uxz2_2 - uxz_2) - 1.5 * uxyz2_2);
    feq_2[14] = tmp_rho_2 * w[14] * (1.0 - 3.0 * (tmp_ux_2 + tmp_uz_2) + 4.5 * (uxz2_2 + uxz_2) - 1.5 * uxyz2_2);
    feq_2[15] = tmp_rho_2 * w[15] * (1.0 + 3.0 * (tmp_uy_2 + tmp_uz_2) + 4.5 * (uyz2_2 + uyz_2) - 1.5 * uxyz2_2);
    feq_2[16] = tmp_rho_2 * w[16] * (1.0 + 3.0 * (tmp_uz_2 - tmp_uy_2) + 4.5 * (uyz2_2 - uyz_2) - 1.5 * uxyz2_2);
    feq_2[17] = tmp_rho_2 * w[17] * (1.0 + 3.0 * (tmp_uy_2 - tmp_uz_2) + 4.5 * (uyz2_2 - uyz_2) - 1.5 * uxyz2_2);
    feq_2[18] = tmp_rho_2 * w[18] * (1.0 - 3.0 * (tmp_uy_2 + tmp_uz_2) + 4.5 * (uyz2_2 + uyz_2) - 1.5 * uxyz2_2);

    if (d_geo[index] == 0)
    {
        for (int q = 0; q < 19; q++)
        {
            d_f_post_1[indexf[q] + Nlattice * q] = d_f_1[index + Nlattice * q] - (d_f_1[index + Nlattice * q] - feq_1[q]) / tauA;
            d_f_post_2[indexf[q] + Nlattice * q] = d_f_2[index + Nlattice * q] - (d_f_2[index + Nlattice * q] - feq_2[q]) / tauB;
        }
    }
    
    if (d_geo[index] == 2)
    {
        for (int q = 0; q < 19; q++)
        {
            if (d_geo[indexf[q]] == 0 || d_geo[indexf[q]] == 2 || d_geo[indexf[q]] == 3 || d_geo[indexf[q]] == 4)
            {
                d_f_post_1[indexf[q] + Nlattice * q] = d_f_1[index + Nlattice * q] - (d_f_1[index + Nlattice * q] - feq_1[q]) / tauA;
                d_f_post_2[indexf[q] + Nlattice * q] = d_f_2[index + Nlattice * q] - (d_f_2[index + Nlattice * q] - feq_2[q]) / tauB;
            }
            else if (d_geo[indexf[q]] == 1)
            {
                if (q == 1)
                {
                    d_f_post_1[index + Nlattice * 2] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 2] = d_f_2[index + Nlattice * q];
                }
                if (q == 2)
                {
                    d_f_post_1[index + Nlattice * 1] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 1] = d_f_2[index + Nlattice * q];                    
                }
                if (q == 3)
                {
                    d_f_post_1[index + Nlattice * 4] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 4] = d_f_2[index + Nlattice * q];
                }
                if (q == 4)
                {
                    d_f_post_1[index + Nlattice * 3] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 3] = d_f_2[index + Nlattice * q];
                }
                if (q == 5)
                {
                    d_f_post_1[index + Nlattice * 6] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 6] = d_f_2[index + Nlattice * q];
                }
                if (q == 6)
                {
                    d_f_post_1[index + Nlattice * 5] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 5] = d_f_2[index + Nlattice * q];
                }
                if (q == 7)
                {
                    d_f_post_1[index + Nlattice * 8] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 8] = d_f_2[index + Nlattice * q];
                }
                if (q == 8)
                {
                    d_f_post_1[index + Nlattice * 7] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 7] = d_f_2[index + Nlattice * q];
                }
                if (q == 9)
                {
                    d_f_post_1[index + Nlattice * 10] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 10] = d_f_2[index + Nlattice * q];
                }
                if (q == 10)
                {
                    d_f_post_1[index + Nlattice * 9] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 9] = d_f_2[index + Nlattice * q];
                }
                if (q == 11)
                {
                    d_f_post_1[index + Nlattice * 12] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 12] = d_f_2[index + Nlattice * q];
                }
                if (q == 12)
                {
                    d_f_post_1[index + Nlattice * 11] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 11] = d_f_2[index + Nlattice * q];
                }
                if (q == 13)
                {
                    d_f_post_1[index + Nlattice * 14] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 14] = d_f_2[index + Nlattice * q];
                }
                if (q == 14)
                {
                    d_f_post_1[index + Nlattice * 13] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 13] = d_f_2[index + Nlattice * q];
                }
                if (q == 15)
                {
                    d_f_post_1[index + Nlattice * 16] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 16] = d_f_2[index + Nlattice * q];
                }
                if (q == 16)
                {
                    d_f_post_1[index + Nlattice * 15] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 15] = d_f_2[index + Nlattice * q];
                }
                if (q == 17)
                {
                    d_f_post_1[index + Nlattice * 18] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 18] = d_f_2[index + Nlattice * q];
                }
                if (q == 18)
                {
                    d_f_post_1[index + Nlattice * 17] = d_f_1[index + Nlattice * q];
                    d_f_post_2[index + Nlattice * 17] = d_f_2[index + Nlattice * q];
                }
            }
            
        }
    }

    if (d_geo[index] == 3)
    {                
        for (int q = 0; q < 19; q++)
        {
            if (q == 0 || q == 1 || q == 2 || q == 3 || q == 4 || q == 5 || q == 7 || q == 8 || q == 9 || q == 11 || q == 13 || q == 14 || q == 16 || q == 18)
            {
                d_f_post_1[indexf[q] + Nlattice * q] = d_f_1[index + Nlattice * q] - (d_f_1[index + Nlattice * q] - feq_1[q]) / tauA;
                d_f_post_2[indexf[q] + Nlattice * q] = d_f_2[index + Nlattice * q] - (d_f_2[index + Nlattice * q] - feq_2[q]) / tauB;
            }
            else 
            {
                d_f_post_1[index + Nlattice * 5] = (-1.0)*((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_in + 2.0 * (feq_1[6] + feq_1[10] + feq_1[12] + feq_1[15] + feq_1[17]) - rho_in) / 3.0 - feq_1[6];
                d_f_post_2[index + Nlattice * 5] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_2[6] + feq_2[10] + feq_2[12] + feq_2[15] + feq_2[17]) - rho_s) / 3.0 - feq_1[6];
                d_f_post_1[index + Nlattice * 9] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_in + 2.0 * (feq_1[6] + feq_1[10] + feq_1[12] + feq_1[15] + feq_1[17]) - rho_in) / 6.0 - feq_1[10];
                d_f_post_2[index + Nlattice * 9] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_2[6] + feq_2[10] + feq_2[12] + feq_2[15] + feq_2[17]) - rho_s) / 6.0 - feq_1[10];
                d_f_post_1[index + Nlattice * 11] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_in + 2.0 * (feq_1[6] + feq_1[10] + feq_1[12] + feq_1[15] + feq_1[17]) - rho_in) / 6.0 - feq_1[12];
                d_f_post_2[index + Nlattice * 11] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_2[6] + feq_2[10] + feq_2[12] + feq_2[15] + feq_2[17]) - rho_s) / 6.0 - feq_1[12];
                d_f_post_1[index + Nlattice * 16] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_in + 2.0 * (feq_1[6] + feq_1[10] + feq_1[12] + feq_1[15] + feq_1[17]) - rho_in) / 6.0 - feq_1[15];
                d_f_post_2[index + Nlattice * 16] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_2[6] + feq_2[10] + feq_2[12] + feq_2[15] + feq_2[17]) - rho_s) / 6.0 - feq_1[15];
                d_f_post_1[index + Nlattice * 18] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_in + 2.0 * (feq_1[6] + feq_1[10] + feq_1[12] + feq_1[15] + feq_1[17]) - rho_in) / 6.0 - feq_1[17];
                d_f_post_2[index + Nlattice * 18] = (-1.0) * ((1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_2[6] + feq_2[10] + feq_2[12] + feq_2[15] + feq_2[17]) - rho_s) / 6.0 - feq_1[17];
            }
        }    
    }

    if (d_geo[index] == 4)
    {
        for (int q = 0; q < 19; q++)
        {
            if (q == 0 || q == 1 || q == 2 || q == 3 || q == 4 || q == 6 || q == 7 || q == 8 || q == 10 || q == 12 || q == 13 || q == 14 || q == 15 || q == 17)
            {
                d_f_post_1[indexf[q] + Nlattice * q] = d_f_1[index + Nlattice * q] - (d_f_1[index + Nlattice * q] - feq_1[q]) / tauA;
                d_f_post_2[indexf[q] + Nlattice * q] = d_f_2[index + Nlattice * q] - (d_f_2[index + Nlattice * q] - feq_2[q]) / tauB;
            }
            else
            {
                d_f_post_1[index + Nlattice * 6] = (-1.0) * (rho_s-(1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_1[5] + feq_1[9] + feq_1[11] + feq_1[16] + feq_1[18])) / 3.0 - feq_1[5];
                d_f_post_2[index + Nlattice * 6] = (-1.0) * (rho_out-(1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_out + 2.0 * (feq_2[5] + feq_2[9] + feq_2[11] + feq_2[16] + feq_2[18])) / 3.0 - feq_1[5];
                d_f_post_1[index + Nlattice * 10] = (-1.0) * (rho_s - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_1[5] + feq_1[9] + feq_1[11] + feq_1[16] + feq_1[18])) / 6.0 - feq_1[9];
                d_f_post_2[index + Nlattice * 10] = (-1.0) * (rho_out - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_out + 2.0 * (feq_2[5] + feq_2[9] + feq_2[11] + feq_2[16] + feq_2[18])) / 6.0 - feq_1[9];
                d_f_post_1[index + Nlattice * 12] = (-1.0) * (rho_s - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_1[5] + feq_1[9] + feq_1[11] + feq_1[16] + feq_1[18])) / 6.0 - feq_1[11];
                d_f_post_2[index + Nlattice * 12] = (-1.0) * (rho_out - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_out + 2.0 * (feq_2[5] + feq_2[9] + feq_2[11] + feq_2[16] + feq_2[18])) / 6.0 - feq_1[11];
                d_f_post_1[index + Nlattice * 15] = (-1.0) * (rho_s - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_1[5] + feq_1[9] + feq_1[11] + feq_1[16] + feq_1[18])) / 6.0 - feq_1[16];
                d_f_post_2[index + Nlattice * 15] = (-1.0) * (rho_out - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_out - 2.0 * (feq_2[5] + feq_2[9] + feq_2[11] + feq_2[16] + feq_2[18])) / 6.0 - feq_1[16];
                d_f_post_1[index + Nlattice * 17] = (-1.0) * (rho_s - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_s + 2.0 * (feq_1[5] + feq_1[9] + feq_1[11] + feq_1[16] + feq_1[18])) / 6.0 - feq_1[18];
                d_f_post_2[index + Nlattice * 17] = (-1.0) * (rho_out - (1.0 / 3 + 4.0 / 18 + 4.0 / 36) * rho_out + 2.0 * (feq_2[5] + feq_2[9] + feq_2[11] + feq_2[16] + feq_2[18])) / 6.0 - feq_1[18];
            }
        }
    }
}

// 0. Main loop

int main()
{
    ofstream outputfile;
    outputfile.open("D:\\rho-test-10.dat");

    const int Nstep = 10;

    // allocate memory on CPU and GPU 
    h_geo = (int*)malloc(sizeof(int) * Nlattice);
    h_ux_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_uy_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_uz_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_ux_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_uy_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_uz_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_rho_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_rho_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_psi_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_psi_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_f_1 = (double*)malloc(sizeof(double) * Nlattice * Q);
    h_f_2 = (double*)malloc(sizeof(double) * Nlattice * Q);
    h_f_post_1 = (double*)malloc(sizeof(double) * Nlattice * Q);
    h_f_post_2 = (double*)malloc(sizeof(double) * Nlattice * Q);
    test = (int*)malloc(sizeof(int) * Nlattice);////////////////////////////////////////////test
    testd = (double*)malloc(sizeof(double) * Nlattice);/////////////////////////////////////test

    cudaMalloc((void**)&d_geo, Nlattice * sizeof(int));
    cudaMalloc((void**)&d_f_1, Nx * Ny * Nz * Q * sizeof(double));
    cudaMalloc((void**)&d_f_2, Nx * Ny * Nz * Q * sizeof(double));
    cudaMalloc((void**)&d_f_post_1, Nx * Ny * Nz * Q * sizeof(double));
    cudaMalloc((void**)&d_f_post_2, Nx * Ny * Nz * Q * sizeof(double));
    cudaMalloc((void**)&d_Fx_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fx_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fy_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fy_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fz_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fz_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_rho_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_rho_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_psi_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_psi_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_ux_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_uy_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_uz_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_ux_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_uy_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_uz_2, Nlattice * sizeof(double));

    pre_geo();
    Initialization(h_geo);

    // initialization on GPU
    cudaMemcpy(d_geo, h_geo, Nx * Ny * Nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_1, h_f_1, Nx * Ny * Nz * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_2, h_f_2, Nx * Ny * Nz * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_post_1, h_f_post_1, Nx * Ny * Nz * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_post_2, h_f_post_2, Nx * Ny * Nz * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_1, h_psi_1, Nlattice * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_2, h_psi_2, Nlattice * sizeof(double), cudaMemcpyHostToDevice);

    for (int step = 0; step < Nstep; step++)
    {
        computeDensity << <grid, block >> > (Nlattice, d_geo, d_f_1, d_f_2, d_rho_1, d_rho_2, d_psi_1, d_psi_2);
        cudaDeviceSynchronize();

        computeSCForces << <grid, block >> > (d_geo, d_psi_1, d_psi_2, d_Fx_1, d_Fy_1, d_Fz_1, d_Fx_2, d_Fy_2, d_Fz_2);
        cudaDeviceSynchronize();

        computeVelocity << <grid, block >> > (d_geo, d_f_1, d_f_2, d_rho_1, d_rho_2, d_Fx_1, d_Fy_1, d_Fz_1, d_Fx_2, d_Fy_2, d_Fz_2, d_ux_1, d_uy_1, d_uz_1, d_ux_2, d_uy_2, d_uz_2);
        cudaDeviceSynchronize();

        StreamBoundary << <grid, block >> > (d_geo, d_f_1, d_f_2, d_f_post_1, d_f_post_2, d_Fx_1, d_Fy_1, d_Fz_1, d_Fx_2, d_Fy_2, d_Fz_2, d_ux_1, d_uy_1, d_uz_1, d_ux_2, d_uy_2, d_uz_2, d_rho_1, d_rho_2);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(testd, d_rho_1, Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)//////////////////////////////////////////////////////test
    {
        for (int y = 0; y < Ny; y++)
        {

            for (int x = 0; x < Nx; x++)
            {
                int k = z * Nx * Ny + y * Nx + x;
                outputfile << x + 1 << "\t" << y + 1 << "\t " << z + 1 << "\t " << testd[k] << endl;
            }
        }
    }
    outputfile.close();

    free(h_geo);
    free(h_f_1);
    free(h_f_2);
    free(h_f_post_1);
    free(h_f_post_2);
    free(h_rho_1);
    free(h_rho_2);
    free(h_psi_1);
    free(h_psi_2);
    free(h_ux_1);
    free(h_uy_1);
    free(h_uz_1);
    free(h_ux_2);
    free(h_uy_2);
    free(h_uz_2);
    free(test);

    cudaFree(d_geo);
    cudaFree(d_f_1);
    cudaFree(d_f_2);
    cudaFree(d_f_post_1);
    cudaFree(d_f_post_2);
    cudaFree(d_rho_1);
    cudaFree(d_rho_2);
    cudaFree(d_psi_1);
    cudaFree(d_psi_2);
    cudaFree(d_ux_1);
    cudaFree(d_uy_1);
    cudaFree(d_uz_1);
    cudaFree(d_ux_2);
    cudaFree(d_uy_2);
    cudaFree(d_uz_2);
    cudaFree(d_Fx_1);
    cudaFree(d_Fy_1);
    cudaFree(d_Fz_1);
    cudaFree(d_Fx_2);
    cudaFree(d_Fy_2);
    cudaFree(d_Fz_2);

    return 0;
}
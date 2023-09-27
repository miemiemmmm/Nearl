#include <iostream>
#include <numeric>
#include <random>

#include "baseutils.h"
#include "icp.h"
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

Eigen::Matrix4d best_fit_transform(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B){
	/*
	Notice:
	1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
	2/ matrix type 'MatrixXd' or 'MatrixXf' matters.
	*/
  Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4,4);
  Eigen::Vector3d centroid_A(0,0,0);
  Eigen::Vector3d centroid_B(0,0,0);
  Eigen::MatrixXd AA = A;
  Eigen::MatrixXd BB = B;
  int row = A.rows();

  for(int i=0; i<row; ++i){
    centroid_A += A.block<1,3>(i,0).transpose();
    centroid_B += B.block<1,3>(i,0).transpose();
  }
  centroid_A /= row;
  centroid_B /= row;
  for(int i=0; i<row; ++i){
    AA.block<1,3>(i,0) = A.block<1,3>(i,0) - centroid_A.transpose();
    BB.block<1,3>(i,0) = B.block<1,3>(i,0) - centroid_B.transpose();
  }

  Eigen::MatrixXd H = AA.transpose()*BB;
  Eigen::MatrixXd U;
  Eigen::VectorXd S;
  Eigen::MatrixXd V;
  Eigen::MatrixXd Vt;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  JacobiSVD<Eigen::MatrixXd> svd(H, ComputeFullU | ComputeFullV);
  U = svd.matrixU();
  S = svd.singularValues();
  V = svd.matrixV();
  Vt = V.transpose();

  R = Vt.transpose()*U.transpose();

  if (R.determinant() < 0 ){
    Vt.block<1,3>(2,0) *= -1;
    R = Vt.transpose()*U.transpose();
  }

  t = centroid_B - R*centroid_A;

  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = t;
  return T;
}

Eigen::Matrix4d _best_fit_transform(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B){
    /*
    Notice:
    1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
    2/ matrix type 'MatrixXd' or 'MatrixXf' matters.
    */
  Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4,4);
  Eigen::Vector3d centroid_A(0,0,0);
  Eigen::Vector3d centroid_B(0,0,0);

  int row_A = A.rows();
  int row_B = B.rows();
  // Compute the centroids of the two point sets
  for(int i=0; i<row_A; ++i){
    centroid_A += A.block<1,3>(i,0).transpose();
  }
  for(int i=0; i<row_B; ++i){
    centroid_B += B.block<1,3>(i,0).transpose();
  }
  centroid_A /= row_A;
  centroid_B /= row_B;

  Eigen::MatrixXd AA(row_A, 3);
  Eigen::MatrixXd BB(row_B, 3);
  for(int i=0; i<row_A; ++i){
    AA.block<1,3>(i,0) = A.block<1,3>(i,0) - centroid_A.transpose();
  }
  for(int i=0; i<row_B; ++i){
    BB.block<1,3>(i,0) = B.block<1,3>(i,0) - centroid_B.transpose();
  }

  Eigen::MatrixXd H = AA.transpose() * BB;
  JacobiSVD<Eigen::MatrixXd> svd(H, ComputeFullU | ComputeFullV);
  Eigen::MatrixXd U = svd.matrixU();
  Eigen::MatrixXd V = svd.matrixV();

  // Compute the rotation matrix
  Eigen::Matrix3d R = V * U.transpose();
  if (R.determinant() < 0) {
    V.col(2) *= -1;
    R = V * U.transpose();
  }

  Eigen::Vector3d t = centroid_B - R*centroid_A;
  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = t;
  return T;

}

Eigen::MatrixXd Matrix4FromCoord(const Eigen::MatrixXd &coord){
	int row = coord.rows();
	Eigen::MatrixXd mat = Eigen::MatrixXd::Ones(4, row);
	for (int i=0; i<row; ++i){
		mat.block<3,1>(0,i) = coord.block<1,3>(i,0).transpose();
	}
	return mat;
}

Eigen::MatrixXd Coord3FromMatrix(const Eigen::MatrixXd &mat){
	int row = mat.rows();
	Eigen::MatrixXd coord = Eigen::MatrixXd::Ones(row,3);
	for(int i=0; i<row; ++i){
		coord.block<1,3>(i,0) = mat.block<1,3>(i,0).transpose();
	}
	return coord;
}

//Eigen::MatrixXd MaskByRows(const Eigen::MatrixXd &mat, const std::vector<int> &mask){
//	int row = mask.size();
//	Eigen::MatrixXd mat_masked = Eigen::MatrixXd::Ones(row,3);
//	for(int i=0; i<row; ++i){
//		mat_masked.block<1,3>(i,0) = mat.block<1,3>(mask[i],0).transpose();
//	}
//	return mat_masked;
//}
//
//Eigen::MatrixXd MaskByCols(const Eigen::MatrixXd &mat, const std::vector<int> &mask){
//	int col = mask.size();
//	Eigen::MatrixXd mat_masked = Eigen::MatrixXd::Ones(3,col);
//	for(int i=0; i<col; ++i){
//		mat_masked.block<3,1>(0,i) = mat.block<3,1>(0,mask[i]);
//	}
//	return mat_masked;
//}

ICP_OUT coarse_to_fine(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const int refinement_rounds){
	// Perform a coarse-to-fine approach to guess the initial transformation
	const int pointnr = std::min(A.rows(), B.rows()), pointnr_A = A.rows(), pointnr_B = B.rows();
	Eigen::MatrixXd A_4d = Matrix4FromCoord(A);
	cout << "A_4d: " << A_4d.rows() << " " << A_4d.cols() << endl;
	Eigen::MatrixXd B_points = Matrix4FromCoord(B);
	Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4,4);
	Eigen::Matrix4d T_final = Eigen::MatrixXd::Identity(4,4);


	vector<int> samples_B = SamplePoints(pointnr_B>>2, 0, pointnr_B-1, true);
	Eigen::MatrixXd B_sample = Eigen::MatrixXd::Ones(pointnr_B>>2, 3);
	for (int idx = 0; idx < (pointnr_B>>2); ++idx){
		B_sample.block<1,3>(idx, 0) = B.block<1,3>(samples_B[idx], 0);
	}

	ICP_OUT result;
	double mean_error = 999, swap_error = 999, max_error_95;

	for (int i = 0; i < refinement_rounds; ++i){
		Eigen::Matrix4d T_tmp = Eigen::MatrixXd::Identity(4,4);
		int num_samples = 500 * (i+1);
		if (num_samples > (pointnr>>1)) num_samples = pointnr >> 1;  // There are not so much points, sample half of them

		// Randomly sample 2 set of non-intersecting points (num_samples) from A for coarse-fine ICP
		vector<int> samples_A = SamplePoints(num_samples, 0, pointnr_A-1, true);
		Eigen::MatrixXd A_sample = Eigen::MatrixXd::Ones(num_samples, 3);
		Eigen::MatrixXd A2_sample = Eigen::MatrixXd::Ones(num_samples, 3);
		cout << "Refinement round " << i+1 << "; Sampled points: " << num_samples << "; Point Number: " << pointnr << "\n";
		// Round 1 ICP
		for (int idx = 0; idx < num_samples; ++idx){
			A_sample.block<1,3>(idx, 0) = A_4d.block<3,1>(0, samples_A[idx]).transpose();
		}
		result = _icp(A_sample, B, 10, 0.000001);
//		A_4d = result.trans * A_4d;
		T_tmp = result.trans * T_tmp;
//		Eigen::Vector3d tmp_trans(10.0, 15.0, 32.0);
//		result.trans.block<3,1>(0,3) = tmp_trans;
		A_4d = result.trans * A_4d;

		// Transform the points and play the round 2 ICP
		for (int idx = 0; idx < num_samples; ++idx){
			A2_sample.block<1,3>(idx, 0) = A_4d.block<3,1>(0, pointnr_A-samples_A[idx]-1).transpose();
		}
		result = _icp(A2_sample, B, 10, 0.000001);
		A_4d = result.trans * A_4d;
		T_tmp = result.trans * T_tmp;

		vector<float> dists = result.distances;
		int index = static_cast<int>(0.5 * dists.size());
		std::nth_element(dists.begin(), dists.begin() + index, dists.end());
		mean_error = std::accumulate(dists.begin(),dists.end(),0.0)/dists.size();
		max_error_95 = dists[index]; // The 95% largest error

		T_final = T_tmp * T_final;
		swap_error = max_error_95;

		if (max_error_95 < swap_error){
			// After this round, the error is smaller than before, so we update the final transformation
			printf("Accepting this round: %d;Mean_error: %f, Swap_error: %f, Max error: %.3f\n", i, mean_error, swap_error, max_error_95);

		} else {
			// This ICP alignment makes the error larger, reject this round's transformation
			printf("Round %d: Error is larger than before, reject this round's transformation\n", i);
			printf("Mean_error: %f, Swap_error: %f, Max error: %.3f \n", mean_error, swap_error, max_error_95);
//			int therand = SamplePoints(1, 0, 6, true)[0];
//			if (therand <= 2 and i != refinement_rounds-1){
//				cout << "Randomize the axis " << therand << endl;
//				T(therand,therand) = -1.0;
//				T_final = T * T_final;
//			}
		}
	}
	result.trans = T_final;
	return result;
}

/*
typedef struct{
    Eigen::Matrix4d trans;
    std::vector<float> distances;
    int iter;
}  ICP_OUT;
*/

ICP_OUT icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations, double tolerance){
  int row = A.rows();
  Eigen::MatrixXd src = Eigen::MatrixXd::Ones(3+1,row);
  Eigen::MatrixXd src3d = Eigen::MatrixXd::Ones(3,row);
  Eigen::MatrixXd dst = Eigen::MatrixXd::Ones(3+1,row);
  NEIGHBOR neighbor;
  Eigen::Matrix4d T;
  Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3,row);
  ICP_OUT result;
  int iter = 0;

  for (int i = 0; i<row; ++i){
    src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
    src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
    dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
  }

  double prev_error = 0;
  double mean_error = 0;
  for (int i=0; i<max_iterations; ++i){
    neighbor = nearest_neighbot(src3d.transpose(),B);

    for(int j=0; j<row; ++j){
      dst_chorder.block<3,1>(0,j) = dst.block<3,1>(0,neighbor.indices[j]);
    }

    T = best_fit_transform(src3d.transpose(),dst_chorder.transpose());

    src = T*src;
    for(int j=0; j<row; ++j){
      src3d.block<3,1>(0,j) = src.block<3,1>(0,j);
    }

    mean_error = std::accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();
    if (abs(prev_error - mean_error) < tolerance){
      break;
    }
    prev_error = mean_error;
    iter = i+2;
  }

  T = best_fit_transform(A,src3d.transpose());
  result.trans = T;
  result.distances = neighbor.distances;
  result.iter = iter;

  return result;
}

ICP_OUT _icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations, double tolerance){
  int row_A = A.rows();
  int row_B = B.rows();
  Eigen::MatrixXd src = Eigen::MatrixXd::Ones(3+1, row_A);
  Eigen::MatrixXd src3d = Eigen::MatrixXd::Ones(3, row_A);
  Eigen::MatrixXd dst = Eigen::MatrixXd::Ones(3+1, row_B);
  NEIGHBOR neighbor;
  Eigen::Matrix4d T;
  Eigen::MatrixXd dst_chorder;
  ICP_OUT result;
  int iter = 0;

  for (int i = 0; i<row_A; ++i){
    src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
    src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
  }

  for (int i = 0; i<row_B; ++i){
    dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
  }


  double prev_error = 0;
  double mean_error = 0;
  for (int i=0; i<max_iterations; ++i){
    neighbor = nearest_neighbot(src3d.transpose(),B);
    Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3, neighbor.indices.size());

    // Fill dst_chorder based on the nearest neighbors found
    for(int j = 0; j < neighbor.indices.size(); ++j){
      dst_chorder.block<3,1>(0,j) = dst.block<3,1>(0, neighbor.indices[j]);
    }
    T = _best_fit_transform(src3d.transpose(),dst_chorder.transpose());
    src = T*src;
    for(int j=0; j<row_A; ++j){
      src3d.block<3,1>(0,j) = src.block<3,1>(0,j);
    }

    mean_error = std::accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();
    if (abs(prev_error - mean_error) < tolerance){
      break;
    }
    prev_error = mean_error;
    iter = i+2;
  }

  T = _best_fit_transform(A,src3d.transpose());
  result.trans = T;
  result.distances = neighbor.distances;
  result.iter = iter;

  return result;
}




/*
typedef struct{
    std::vector<float> distances;
    std::vector<int> indices;
} NEIGHBOR;
*/

NEIGHBOR nearest_neighbot(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst){
    int row_src = src.rows();
    int row_dst = dst.rows();
    Eigen::Vector3d vec_src;
    Eigen::Vector3d vec_dst;
    NEIGHBOR neigh;
    float min = 100;
    int index = 0;
    float dist_temp = 0;

    for(int ii=0; ii < row_src; ++ii){
        vec_src = src.block<1,3>(ii,0).transpose();
        min = 100;
        index = 0;
        dist_temp = 0;
        for(int jj=0; jj < row_dst; ++jj){
            vec_dst = dst.block<1,3>(jj,0).transpose();
            dist_temp = dist(vec_src,vec_dst);
            if (dist_temp < min){
                min = dist_temp;
                index = jj;
            }
        }
        // cout << min << " " << index << endl;
        // neigh.distances[ii] = min;
        // neigh.indices[ii] = index;
        neigh.distances.push_back(min);
        neigh.indices.push_back(index);
    }
    return neigh;
}


float dist(const Eigen::Vector3d &pta, const Eigen::Vector3d &ptb){
    return sqrt((pta[0]-ptb[0])*(pta[0]-ptb[0]) + (pta[1]-ptb[1])*(pta[1]-ptb[1]) + (pta[2]-ptb[2])*(pta[2]-ptb[2]));
}

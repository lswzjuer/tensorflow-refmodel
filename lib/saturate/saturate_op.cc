/*
Reference Model
Copyright (c) 2019 MobaiTech Inc 
Author: Abinash Mohanty
*/

#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Sp")
	.Attr("T: {float, double}")
	.Attr("bw: int")	// Bit width value, currently supported, 8bit and 16bit	
	.Input("bottom_data: T")
	.Output("top_data: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

template <typename Device, typename T>
class SaturateOp : public OpKernel {
	public:
		explicit SaturateOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("bw", &bw_));
			OP_REQUIRES(context, bw_>= 0, errors::InvalidArgument("Need bw >= 0, got ",bw_));
		}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();

	T max_val;
	T min_val;

	if ( bw_ == 8 ){ 	// 8 bits saturation
		max_val = (T)127.0;		
		min_val = (T)-128.0;
	}
	else {		// 16 bit saturation	
		max_val = (T)32767;
		min_val = (T)-32768.0;	
	}

    const int N = input.size();
    for (int i = 0; i < N; i++) {
		if (input(i) > max_val)
			output_flat(i) = max_val;
		else if (input(i) < min_val)
			output_flat(i) = min_val;
		else
			output_flat(i) = input(i);
    }
  }
  private:
    int bw_;
};

//REGISTER_KERNEL_BUILDER(Name("Lp").Device(DEVICE_CPU), LpOp);
REGISTER_KERNEL_BUILDER(Name("Sp").Device(DEVICE_CPU).TypeConstraint<float>("T"), SaturateOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Sp").Device(DEVICE_CPU).TypeConstraint<double>("T"), SaturateOp<CPUDevice, double>);

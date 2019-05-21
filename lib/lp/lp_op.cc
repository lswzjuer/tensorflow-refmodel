/*
Reference Model
Copyright (c) 2018 MobaiTech Inc 
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

REGISTER_OP("Lp")
	.Attr("T: {float, double}")
	.Attr("bw: int")	// Bit width value, currently supported, 8bit and 16bit	
	.Attr("rs: int")	// Right shift value
	.Input("bottom_data: T")
	.Output("top_data: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

template <typename Device, typename T>
class LpOp : public OpKernel {
	public:
		explicit LpOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("bw", &bw_));
			OP_REQUIRES(context, bw_>= 0, errors::InvalidArgument("Need bw >= 0, got ",bw_));
			OP_REQUIRES_OK(context, context->GetAttr("rs", &rs_));
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

    //float temp_scale = rs_ > 0 ? (1 << rs_) : 1.0f / (1 << (-rs_));

    const int N = input.size();
    for (int i = 0; i < N; i++) {

		float input_shifted = input(i)/pow(2,rs_);
		int   input_rounded = (input_shifted > 0.0) ? (input_shifted + 0.5) : (input_shifted - 0.5);

		if (input_rounded <= int(min_val))	// Saturate to max value
			output_flat(i) = min_val;			
		else if (input_rounded >= int(max_val))	// Saturate to min value
			output_flat(i) = max_val;
		else
			output_flat(i) = (T)input_rounded;		// bit slice and send out

	//	int shifted_out = int(input(i)) >> rs_;	// Shift out rs bits 
	//	int sliced_out = shifted_out % int(pow(2.0,bw_-1));	// slice bitwidth -1 bits from lsb (1 bit for sign)

	//	if (shifted_out <= int(min_val))	// Saturate to max value
	//		output_flat(i) = min_val;			
	//	else if (shifted_out >= int(max_val))	// Saturate to min value
	//		output_flat(i) = max_val;
	//	else
	//		output_flat(i) = (T)sliced_out;		// bit slice and send out
    }
  }
  private:
    int bw_;
    int rs_; 	
};

//REGISTER_KERNEL_BUILDER(Name("Lp").Device(DEVICE_CPU), LpOp);
REGISTER_KERNEL_BUILDER(Name("Lp").Device(DEVICE_CPU).TypeConstraint<float>("T"), LpOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Lp").Device(DEVICE_CPU).TypeConstraint<double>("T"), LpOp<CPUDevice, double>);

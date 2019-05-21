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

REGISTER_OP("Fp")
	.Attr("T: {float, double}")
	.Attr("fl: int")	// fractional length value
	.Input("bottom_data: T")
	.Output("top_data: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

template <typename Device, typename T>
class FpOp : public OpKernel {
	public:
		explicit FpOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("fl", &fl_));
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

    const int N = input.size();
    for (int i = 0; i < N; i++) {
	
		// Deprecated implementation.
    	output_flat(i) = (T)input(i)/(T)pow(2,fl_);

    }
  }
  private:
    int fl_;
};

REGISTER_KERNEL_BUILDER(Name("Fp").Device(DEVICE_CPU).TypeConstraint<float>("T"), FpOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Fp").Device(DEVICE_CPU).TypeConstraint<double>("T"), FpOp<CPUDevice, double>);

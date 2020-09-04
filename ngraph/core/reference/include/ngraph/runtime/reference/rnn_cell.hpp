//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <ngraph/runtime/reference/add.hpp>
#include <ngraph/runtime/reference/clamp.hpp>
#include <ngraph/runtime/reference/matmul.hpp>
#include <ngraph/runtime/reference/relu.hpp>
#include <ngraph/runtime/reference/sigmoid.hpp>
#include <ngraph/runtime/reference/tanh.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void rnn_cell(const T* X,
                          const Shape& X_shape,
                          const T* H,
                          const Shape& H_shape,
                          const T* W,
                          const Shape& W_shape,
                          const T* R,
                          const Shape& R_shape,
                          const T* B,
                          const Shape& B_shape,
                          T* dst_data,
                          const std::string& activation_f,
                          float clip)
            {
                // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
                // The names used below are analogous to the one used in ONNX documentation.
                //
                // ------ ACRONYMS ------
                // i_t - input gate at current time step
                // t - time step (t-1 means previous time step)
                // X   - The input data tensor. Shape: [batch_size, input_size].
                // W   - The weight tensor for input gate. Shape: [hidden_size, input_size].
                // R   - The recurrence weight tensor for input gate. Shape: [hidden_size,
                // hidden_size].
                // H_t - The hidden state tensor at current time step. Shape: [batch_size,
                // hidden_size].
                // B   - The bias tensor for the input gate. Shape: [hidden_size].
                // Wb  - W bias vectors for input gate.
                // Rb  - R bias vectors for input gate.
                // ------ VARIABLE NAMES ------
                // Xt_W    - Input sequence multiplied by weights tensor at current time step.
                // Ht_R    - Hidden state multiplied by weights tensor at current time step.

                // (.) - Denotes element-wise multiplication.
                // *   - Denotes dot product.

                // ---- Equations ----
                // f - is activation functions.
                // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                // --------------------

                // Xt*(W^T)
                std::vector<T> Xt_W(X_shape[0] * W_shape[0]);
                reference::matmul(
                    X, W, Xt_W.data(), X_shape, W_shape, {X_shape[0], W_shape[0]}, false, true);

                // Ht-1*(R^T)
                std::vector<T> Ht_R(H_shape[0] * R_shape[0]);
                reference::matmul(
                    H, R, Ht_R.data(), H_shape, R_shape, {H_shape[0], R_shape[0]}, false, true);

                // Ht-1*(R^T) + Wb + Rb
                std::vector<T> Ht_R_B(H_shape[0] * R_shape[0]);
                reference::add(Ht_R.data(),
                               B,
                               Ht_R_B.data(),
                               {H_shape[0], R_shape[0]},
                               B_shape,
                               op::AutoBroadcastSpec::NUMPY);

                // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
                std::vector<T> i_t(H_shape[0] * R_shape[0]);
                reference::add(Xt_W.data(),
                               Ht_R_B.data(),
                               i_t.data(),
                               {X_shape[0], W_shape[0]},
                               {H_shape[0], R_shape[0]},
                               op::AutoBroadcastSpec::NUMPY);

                // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                if (clip != 0.f)
                {
                    reference::clamp(i_t.data(),
                                     i_t.data(),
                                     static_cast<T>(-clip),
                                     static_cast<T>(clip),
                                     i_t.size());
                }
                if (activation_f == "relu")
                {
                    reference::relu(i_t.data(), dst_data, i_t.size());
                }
                else if (activation_f == "sigmoid")
                {
                    reference::sigmoid(i_t.data(), dst_data, i_t.size());
                }
                else if (activation_f == "tanh")
                {
                    reference::tanh(i_t.data(), dst_data, i_t.size());
                }
                else
                {
                    throw ngraph_error("Activation function " + activation_f +
                                       " is not supported.");
                }
            }
        }
    }
}
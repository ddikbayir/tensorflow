/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/placer.h"

#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <cstring>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/kernel_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/port.h"

#include <fstream>
using namespace std;
namespace tensorflow {

namespace {

// Returns true if the node has no inputs and produces outputs
// that are consumed by a single node.
//
// TODO(vrv): Currently this handles only nodes with one output, but
// this could be extended to handle the case where a node has many
// outputs that are connected to nodes in the same colocation group.
/*
bool IsGeneratorNode(const Node* node) {
  return node->num_inputs() == 0 && node->num_outputs() == 1 &&
         !IsRefType(node->output_type(0));
}
*/
void LogDeviceAssignment(const Node* node, bool log_device_placement) {
  // Log placement if log_device_placement is set.
  if (log_device_placement) {
    printf("%s: (%s): %s\n", node->name().c_str(), node->type_string().c_str(),
           node->assigned_device_name().c_str());
    LOG(INFO) << node->name() << ": "
              << "(" << node->type_string() << ")"
              << node->assigned_device_name();
  }
}

Status AssignAndLog(int assigned_device, Node* node,
                    ColocationGraph* colocation_graph,
                    bool log_device_placement) {
  node->set_assigned_device_name_index(assigned_device);

  // Constraint the group of node to the assigned device.
  //We disable this to disable placer algo (DOGA)
  //TF_RETURN_IF_ERROR(colocation_graph->LimitToAssignedDevice(*node));

  LogDeviceAssignment(node, log_device_placement);
  return Status::OK();
}

}  // namespace

Placer::Placer(Graph* graph, const string& function_name,
               const FunctionLibraryDefinition* flib_def,
               const DeviceSet* devices, const Device* default_device,
               bool allow_soft_placement, bool log_device_placement)
    : graph_(graph),
      function_name_(function_name),
      flib_def_(flib_def),
      devices_(devices),
      default_device_(default_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {}

Placer::Placer(Graph* graph, const string& function_name,
               const DeviceSet* devices, const Device* default_device)
    : Placer(graph, function_name, &graph->flib_def(), devices, default_device,
             true, false) {}

Placer::Placer(Graph* graph, const string& function_name,
               const DeviceSet* devices)
    : Placer(graph, function_name, &graph->flib_def(), devices, nullptr, true,
             false) {}

Placer::~Placer() {}

/**
/* Helper function to identify if the target operation has a GPU implementation
*/
int HasGPUKernel(Node* node) {
  KernelList krns = GetRegisteredKernelsForOp(node->type_string());
  int res = 0;
  AttrSlice as = node->attrs();
  cout << "DEBUG:" << SummarizeAttrs(node->def()) << endl;
  string attr_sum = SummarizeAttrs(node->def());
  
  char * cstr = new char [attr_sum.length()+1 ];
  strcpy(cstr, attr_sum.c_str());

  bool match = false;
  for(const auto& kernel_def : krns.kernel()) {
    if(kernel_def.device_type().compare("GPU") == 0) {
       KernelAttrsMatch(kernel_def, as, &match);
      if(match) {
        res = 1;
        break;
      }
    } 
  }
  return res;
}
//DOGA: Alternate placer for custom placement
Status Placer::Run() {
  if (devices_->devices().empty()) {
    return errors::FailedPrecondition("No devices are registered");
  }

  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("placer_input", *graph_, nullptr);
  }
  if (VLOG_IS_ON(5)) {
    for (const Node* node : graph_->op_nodes()) {
      VLOG(5) << "    " << node->name() << ": requested: '"
              << node->requested_device() << "' assigned: '"
              << node->assigned_device_name() << "'";
    }
  }
  FunctionStack stack(function_name_);
  ColocationGraph colocation_graph(graph_, stack, flib_def_, devices_,
                                   default_device_, allow_soft_placement_,
                                   log_device_placement_);


  TF_RETURN_IF_ERROR(colocation_graph.Initialize());
  cout << "Reading placement file..." << endl; 
  std::unordered_map<string, int> parcore_placement;
  
  
  //char dest_path[200];
  
  std::string homepath = getenv("HOME");
  
  //cout << "HOME-DOGA" << (homepath + "/placement.place").c_str() << endl;
  ifstream placement_file((homepath + "/placement.place").c_str());
  
  //strcat(dest_path, homedir);
  //strcat(dest_path, "/placement.place");
  //strcat(homedir,"/placement.place");
  //cout << dest_path << endl; 
  //std::ifstream placement_file(dest_path);
  //cout << "Read placement file..." << endl;

  string node_name;
  int placed_dev;
  int counter= 0;
  //map nodes to assigned devices
  while(placement_file >> node_name >> placed_dev) {
    //cout << "DEBUG-DOGA" << endl;
    //cout << node_name << endl;
    parcore_placement[node_name] = placed_dev;
  }
  //cout << "Populated unordered placement map..." << endl;
  std::vector<Device*> devs = devices_->devices();
  for(Node* node : graph_->op_nodes()) {
    int assigned_device = 0;
    int assigned_dev_id = 0;
    
    //Assign the node
    assigned_dev_id = parcore_placement[node->name()] + 1; // -1 for cpu
    //cout << "Assigned Dev ID: " << assigned_dev_id << endl;
    //cout << "Node: " << node->name() << endl;
    //cout << "Node ID: " << assigned_dev_id << endl;
    
    //cout << "Number of devices in the system: " << devs.size() << endl;
    assigned_device = graph_->InternDeviceName(devs[assigned_dev_id]->name());
    //cout << "Assigned device created: " << assigned_device << endl;
    TF_RETURN_IF_ERROR(AssignAndLog(assigned_device, node, &colocation_graph, log_device_placement_)); 
  }
  //cout << "Finished device assignment without any errors..." << endl;
  return Status::OK();
}


bool Placer::CanAssignToDevice(const string& candidate_device_name,
                               const std::vector<Device*>& devices) const {
  if (!candidate_device_name.empty()) {
    // 'devices' lists the set of devices that the placer or the user has
    // constrained the operation to.  "candidate_device_name" must
    // refer to a concrete Device that is in the list of 'devices'.
    const Device* other_device =
        devices_->FindDeviceByName(candidate_device_name);
    if (std::find(devices.begin(), devices.end(), other_device) !=
        devices.end()) {
      return true;
    }
  }

  return false;
}

}  // namespace tensorflow

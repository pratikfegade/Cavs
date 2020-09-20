#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace std::chrono;

// Gflags
static bool IsNonEmptyMessage(const char *flagname, const std::string &value) {
  return value[0] != '\0';
}

// SST
int SST_MAX_LEN = 56;
int SST_MAX_DEPENDENCY = 111;
int SST_NUM_SAMPLES = 8544;

class SSTReader {
 public:
  SSTReader(const string input, const string graph) :
    input_path(input), graph_path(graph),
    input_file(input), graph_file(graph) {
  }

    void next_batch(vector<int>* batch_graph, vector<float>* batch_input, int* num_nodes) {
    std::fill(batch_input->begin(), batch_input->end(), 0);
    std::fill(batch_graph->begin(), batch_graph->end(), -1);
    int i = 0;
    *num_nodes = 0;
    while (i < FLAGS_batch_size) {
      if (input_file.eof()) { // which mean it reaches the end of the file
        input_file.clear();
        input_file.seekg(0, ios::beg);
        graph_file.clear();
        graph_file.seekg(0, ios::beg);
      }

      string input_str, graph_str;
      getline(input_file, input_str);
      if (input_str.length() > 0) {
        getline(graph_file, graph_str);
        int length;
        process_graph<int>(batch_graph->data() + i*SST_MAX_DEPENDENCY, &length, graph_str);
        CHECK(SST_MAX_DEPENDENCY >= length);
	(*num_nodes) += length;

        process_data<float>(batch_input->data() + i*SST_MAX_DEPENDENCY, &length, input_str);
        CHECK(SST_MAX_LEN >= length);
        i++;
      }
    }
  }

 private:
  template<typename T>
  void process_data(T* data, int* len, const string& str) {
    stringstream input_stream(str);
    int val, idx = 0;
    while (input_stream >> val) {
      data[idx] = val;
      idx++;
    }
    *len = idx;
  }

  template<typename T>
  void process_graph(T* data, int* len, const string& str) {
    stringstream input_stream(str);
    int val, idx = 0;
    while (input_stream >> val) {
      // data[idx] = val-1;
      data[idx] = val;
      idx++;
    }
    *len = idx;
    CHECK((data[idx-1] == -1));
  }

  string input_path;
  string graph_path;

  ifstream input_file;
  ifstream graph_file;
};


// Measurement
float measure_time(std::function<float()> runner, bool mem_profile = false) {
  int w_iters = mem_profile ? 0 : 10;
  int a_iters = mem_profile ? 1 : 10;
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  float cg_gen_time = 0.0;
  float cg_exe_time = 0.0;
  for (int i = 0; i < a_iters; ++i) {
    auto p = runner();
    cg_gen_time += p.first;
    cg_exe_time += p.second;
  }
  return cg_exe_time / a_iters;
}

void report_time(float all_time_us, int num_nodes, int num_batches) {
  float all_time_ms = all_time_us / 1000.0;

  float node_time_us = all_time_us / num_nodes;
  float batch_time_ms = all_time_ms / num_batches;
  std::cout << "RESULTS," << node_time_us << "," << batch_time_ms << std::endl;
  std::cout << "MEM," << dynet::get_max_mem_usage() << std::endl;
}

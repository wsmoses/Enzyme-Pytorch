#include <torch/extension.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

#include <dlfcn.h>
std::function<void(void*, size_t, void*)> compile(std::string filename, std::string function) {
    char buffer [L_tmpnam];
    tmpnam (buffer);
    int res;
    char data[1024];
    sprintf(data, "clang++ %s -O3 -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -Xclang -new-struct-path-tbaa -S -emit-llvm -o %s.ll", filename.c_str(), buffer);
    printf("running compile - %s\n", data);
    res = system(data);
    printf("ran compile - %s\n", data);
    assert(res == 0);

    printf("making buffer 2\n");

    char buffer2 [L_tmpnam];
    printf("making tm buffer 2\n");
    tmpnam (buffer2);
    printf("made buffer 2\n");

    sprintf(data, "clang++ -fPIC -shared %s.ll -o %s.so", buffer, buffer2);
    printf("running library - %s\n", data);
    res = system(data);
    printf("ran library - %s\n", data);
    assert(res == 0);

    char buffer3[L_tmpnam];
    sprintf(buffer3, "%s.so", buffer2);

    printf("running dlopen\n");
    void* lib = dlopen(buffer3, RTLD_LAZY);
    assert(lib);
    printf("running dlsym\n");
    void* sym = dlsym(lib, function.c_str());
    assert(sym);
    auto f = (void(*)(void*, size_t, void*))sym;
    return f;
}

std::function<void(void*, void*, size_t, void*)> diffecompile(std::string filename, std::string function) {
    int res;

    char buffer [L_tmpnam];
    tmpnam (buffer);
    char data[1024];
    sprintf(data, "clang++ -O3 %s -DTF_ENZYME=1 -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -Xclang -new-struct-path-tbaa -S -emit-llvm -o %s.ll", filename.c_str(), buffer);
    printf("running compile - %s\n", data);
    res = system(data);
    printf("ran compile - %s\n", data);
    assert(res == 0);

    sprintf(data, "~/git/Enzyme/build/bin/opt %s.ll -load=%s -S -enzyme -mem2reg -instcombine -simplifycfg -adce -loop-deletion -simplifycfg -o %s.ll", buffer, "/home/wmoses/git/Enzyme/enzyme/build-dbg/Enzyme/LLVMEnzyme-7.so", buffer);
    //sprintf(data, "~/git/Enzyme/build/bin/opt %s.ll -load=%s -S -enzyme -O3 -o %s.ll", buffer, "/home/wmoses/git/Enzyme/enzyme/build-dbg/Enzyme/LLVMEnzyme-7.so", buffer);
    printf("running compile - %s\n", data);
    res = system(data);
    printf("ran compile - %s\n", data);
    assert(res == 0);


    printf("making buffer 2\n");

    char buffer2 [L_tmpnam];
    printf("making tm buffer 2\n");
    tmpnam (buffer2);
    printf("made buffer 2\n");

    sprintf(data, "clang++ -fPIC -shared %s.ll -o %s.so", buffer, buffer2);
    printf("running library - %s\n", data);
    res = system(data);
    printf("ran library - %s\n", data);
    assert(res == 0);

    char buffer3[L_tmpnam];
    sprintf(buffer3, "%s.so", buffer2);

    printf("running dlopen\n");
    void* lib = dlopen(buffer3, RTLD_LAZY);
    assert(lib);
    std::string tofind = "diffe" + function;
    printf("running dlsym %s\n", tofind.c_str());
    void* sym = dlsym(lib, tofind.c_str());
    assert(sym);
    auto diffef = (void(*)(void*, void*, size_t, void*))sym;
    return diffef;
}

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input, std::string filename, std::string function) {

  if(!input.is_contiguous()) {
    std::cout << "input not is_contiguous\n";
  }
  torch::Tensor result = at::empty(input.sizes(), input.options());


  auto f = compile(filename, function);


  f(input.data<float>(), input.numel(), result.data<float>());

  return {result};
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_out,
    torch::Tensor input, std::string filename, std::string function) {
  torch::Tensor inpp = at::empty(input.sizes(), input.options());

  if(!input.is_contiguous()) {
    std::cout << "input not is_contiguous\n";
  }
  if(!grad_out.is_contiguous()) {
    std::cout << "grad_out not is_contiguous\n";
  }
  auto df = diffecompile(filename, function);

  df(input.data<float>(), inpp.data<float>(), input.numel(), grad_out.data<float>());

  return {inpp};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}

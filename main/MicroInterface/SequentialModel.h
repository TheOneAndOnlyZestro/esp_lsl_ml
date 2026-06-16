#ifndef SEQMODEL_H
#define SEQMODEL_H

#include "Model.h"
class SequentialModel{
    private:
        tflite::Model* tflite_model = nullptr;
        tflite::MicroInterpreter* interpreter = nullptr;
    public:
};

#endif // SEQMODEL_H
void CosineSimilarityBatch(const float *vecs1, const float *vecs2, float *result, int rows1, int rows2, int cols);
void CosineSimilarityQueryMax(const float *vecs1, const float *vecs2, unsigned int *ret_indexes, unsigned int rows1, unsigned int rows2, unsigned int cols);
void CosineSimilarityQueryMin(const float *vecs1, const float *vecs2, unsigned int *ret_indexes, unsigned int rows1, unsigned int rows2, unsigned int cols);


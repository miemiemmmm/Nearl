//#include <vector>
#include "contourdata.h"


template <class Data_Type>
class Block_Array{
public:
    Block_Array(int block_size) : block_size(block_size), ae(0), anxt(0), ale(0), alsize(0), afsize(0), a(NULL), alist(NULL) {}

    ~Block_Array() {
        for (int k = 0; k < ale; ++k)
            delete[] alist[k];
        delete[] alist;
    }

    int size() { return ae + afsize; }

    Data_Type element(int k) {
        return (k >= afsize ? a[k - afsize] : alist[k / block_size][k % block_size]);
    }

    void set_element(int k, Data_Type e) {
        if (k >= afsize) a[k - afsize] = e;
        else alist[k / block_size][k % block_size] = e;
    }

    void add_element(T e) {
        if (ae == block_size || alsize == 0) next_block();
        a[ae++] = e;
    }

    void array(Data_Type* carray) {
        int k = 0;
        for (int i = 0; i + 1 < anxt; ++i) {
            Data_Type* b = alist[i];
            for (int j = 0; j < block_size; ++j, ++k)
                carray[k] = b[j];
        }
        for (int j = 0; j < ae; ++j, ++k)
            carray[k] = a[j];
    }

    void reset() {
        ae = anxt = afsize = 0;
        a = (alist ? alist[0] : a);
    }

private:
    int block_size;
    int ae;
    int anxt;
    int ale;
    int alsize;
    int afsize;
    Data_Type* a;
    Data_Type** alist;

    void next_block() {
        if (anxt >= ale) {
            if (alsize == 0) {
                alsize = 1024;
                alist = new Data_Type* [alsize];
            }
            if (ale == alsize) {
                Data_Type** alist2 = new Data_Type* [2 * alsize];
                for (int k = 0; k < alsize; ++k)
                    alist2[k] = alist[k];
                delete[] alist;
                alist = alist2;
                alsize *= 2;
            }
            alist[ale++] = new Data_Type[block_size];
        }
        a = alist[anxt++];
        ae = 0;
        afsize = (anxt - 1) * block_size;
    }
};
class Grid_Cell{
public:
  int k0, k1;  // Cell position in xy plane.
  int vertex[20];  // Vertex numbers for 12 edges and 8 corners.
  bool boundary;  // Contour reaches boundary.
};

class GridCellList{
public:
  GridCellList(int size0, int size1) : cells(CONTOUR_ARRAY_BLOCK_SIZE){
    this->cell_table_size0 = size0+2;  // Pad by one grid cell.
    int cell_table_size1 = size1+2;
    int size = cell_table_size0 * cell_table_size1;
    this->cell_count = 0;
    this->cell_base_index = 2;
    this->cell_table = new int[size];
    for (int i = 0 ; i < size ; ++i)
      cell_table[i] = no_cell;
    for (int i = 0 ; i < cell_table_size0 ; ++i)
      cell_table[i] = cell_table[size-i-1] = out_of_bounds;
    for (int i = 0 ; i < size ; i += cell_table_size0)
      cell_table[i] = cell_table[i+cell_table_size0-1] = out_of_bounds;
  }
  ~GridCellList(){
    delete_cells();
    delete [] cell_table;
  }
  void set_edge_vertex(int k0, int k1, Edge_Number e, int v){
    GridCell *c = cell(k0,k1);
    if (c)
      c->vertex[e] = v;
  }
  void set_corner_vertex(int k0, int k1, Corner_Number corner, int v){
    GridCell *c = cell(k0,k1);
    if (c){
        c->vertex[12+corner] = v;
        c->boundary = true;
    }
  }
  void finished_plane(){
    cell_base_index += cell_count;
    cell_count = 0;
  }
  int cell_count;    // Number of elements of cells currently in use.
  BlockArray<GridCell*> cells;

private:
  static const int out_of_bounds = 0;
  static const int no_cell = 1;
  int cell_table_size0;
  int cell_base_index;  // Minimum valid cell index.
  int *cell_table;      // Maps cell plane index to cell list index.

  // Get cell, initializing or allocating a new one if necessary.
  GridCell *cell(int k0, int k1){
    int i = k0+1 + (k1+1)*cell_table_size0;
    int c = cell_table[i];
    if (c == out_of_bounds)
      return NULL;

    GridCell *cp;
    if (c != no_cell && c >= cell_base_index){
      cp = cells.element(c-cell_base_index);
    } else {
      cell_table[i] = cell_base_index + cell_count;
      if (cell_count < cells.size()){
        cp = cells.element(cell_count);
      } else {
        cells.add_element(cp = new GridCell);
      }
      cp->k0 = k0;
      cp->k1 = k1;
      cp->boundary = false;
      cell_count += 1;
    }
    return cp;
  }

  void delete_cells(){
    int cc = cells.size();
    for (int c = 0 ; c < cc ; ++c){delete cells.element(c);}

  }
};



class Contour_Surface
{
public:
  virtual ~Contour_Surface() {}
  virtual int vertex_count() = 0;
  virtual int triangle_count() = 0;
  virtual void geometry(float *vertex_xyz, int *triangle_vertex_indices) = 0;
  virtual void normals(float *normals) = 0;
};

template <class Data_Type>
class CSurface : public Contour_Surface
{
public:
  CSurface(const Data_Type *grid, const int size[3], const int stride[3],
           float threshold, bool cap_faces, int block_size)
    : grid(grid), threshold(threshold), cap_faces(cap_faces),
      vxyz(3*block_size), tvi(3*block_size)
    {
      for (int a = 0 ; a < 3 ; ++a){
        this->size[a] = size[a];
        this->stride[a] = stride[a];
      }
      compute_surface();
    }

  virtual ~CSurface() {}

  virtual int vertex_count() override { return vxyz.size()/3; }
  virtual int triangle_count() override { return tvi.size()/3; }
  virtual void geometry(float *vertex_xyz, int *triangle_vertex_indices) override
  {
    // implementation here
    vxyz.array(vertex_xyz);
    tvi.array(triangle_vertex_indices);
  }
  virtual void normals(float *normals);

private:
  const Data_Type *grid;
  int size[3];
  int stride[3];
  float threshold;
  bool cap_faces;
  Block_Array<float> vxyz;
  Block_Array<int> tvi;

  void compute_surface()
  int add_cap_vertex_l0(int bv, int k0, int k1, int k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  int add_cap_vertex_r0(int bv, int k0, int k1, int k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  int add_cap_vertex_l1(int bv, int k0, int k1, int k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  int add_cap_vertex_r1(int bv, int k0, int k1, int k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  int add_cap_vertex_l2(int bv, int k0, int k1, int k2,
			  Grid_Cell_List &gp1);
  int add_cap_vertex_r2(int bv, int k0, int k1, int k2,
			  Grid_Cell_List &gp0);
  void make_triangles(Grid_Cell_List &gp0, int k2);
  void add_triangle_corner(int v) { tvi.add_element(v); }
  int create_vertex(float x, float y, float z){
    vxyz.add_element(x);
    vxyz.add_element(y);
    vxyz.add_element(z);
    return vertex_count()-1;
  }
  void make_cap_triangles(int face, int bits, int *cell_vertices){
    int fbits = face_corner_bits[face][bits];
    int *t = cap_triangle_table[face][fbits];
    for (int v = *t ; v != -1 ; ++t, v = *t)
	add_triangle_corner(cell_vertices[v]);
  }
};

template <class Data_Type>
void CSurface<Data_Type>::compute_surface()
{
  // Check if grid point value is above threshold. If so, check if six connected edges
  // cross the contour surface and make vertex, add vertex to four bordering
  // grid cells, and triangulate grid cells between two z grid planes.
  Grid_Cell_List gcp0(size[0]-1, size[1]-1), gcp1(size[0]-1, size[1]-1);
  for (int k2 = 0 ; k2 < size[2] ; ++k2){
    Grid_Cell_List &gp0 = (k2%2 ? gcp1 : gcp0), &gp1 = (k2%2 ? gcp0 : gcp1);
    mark_plane_edge_cuts(gp0, gp1, k2);
    if (k2 > 0){
      make_triangles(gp0, k2);  // Create triangles for cell plane.
    }
    gp0.finished_plane();
  }
}

template <class Data_Type>
void CSurface<Data_Type>::make_triangles(Grid_Cell_List &gp0, int k2){
  int step0 = stride[0], step1 = stride[1], step2 = stride[2];
  int k0_size = size[0], k1_size = size[1], k2_size = size[2];
  Block_Array<Grid_Cell *> &clist = gp0.cells;
  int cc = gp0.cell_count;
  const Data_Type *g0 = grid + step2*(int)(k2-1);
  int step01 = step0 + step1;
  for (int k = 0 ; k < cc ; ++k){
    Grid_Cell *c = clist.element(k);
    const Data_Type *gc = g0 + step0*(int)c->k0 + step1*(int)c->k1, *gc2 = gc + step2;
    int bits = ((gc[0] < threshold ? 0 : 1) |
                (gc[step0] < threshold ? 0 : 2) |
                (gc[step01] < threshold ? 0 : 4) |
                (gc[step1] < threshold ? 0 : 8) |
                (gc2[0] < threshold ? 0 : 16) |
                (gc2[step0] < threshold ? 0 : 32) |
                (gc2[step01] < threshold ? 0 : 64) |
                (gc2[step1] < threshold ? 0 : 128));

    int *cell_vertices = c->vertex;
    int *t = triangle_table[bits];
    for (int e = *t ; e != -1 ; ++t, e = *t){add_triangle_corner(cell_vertices[e]);};

    if (c->boundary && cap_faces){
      // Check 6 faces for being on boundary, assemble 4 bits for
      // face and call triangle building routine.
      if (c->k0 == 0)
        make_cap_triangles(4, bits, cell_vertices);
      if (c->k0 + 2 == k0_size)
        make_cap_triangles(2, bits, cell_vertices);
      if (c->k1 == 0)
        make_cap_triangles(1, bits, cell_vertices);
      if (c->k1 + 2 == k1_size)
        make_cap_triangles(3, bits, cell_vertices);
      if (k2 == 1)
        make_cap_triangles(0, bits, cell_vertices);
      if (k2 + 1 == k2_size)
        make_cap_triangles(5, bits, cell_vertices);
    }
  }
}

template <class Data_Type>
void CSurface<Data_Type>::mark_plane_edge_cuts(Grid_Cell_List &gp0, Grid_Cell_List &gp1, int k2){
  int k0_size = size[0], k1_size = size[1], k2_size = size[2];
  for (int k1 = 0 ; k1 < k1_size ; ++k1){
    if (k1 == 0 || k1+1 == k1_size || k2 == 0 || k2+1 == k2_size){
      for (int k0 = 0 ; k0 < k0_size ; ++k0){
        mark_boundary_edge_cuts(k0, k1, k2, gp0, gp1);
      }
    } else {
      if (k0_size > 0){
        mark_boundary_edge_cuts(0, k1, k2, gp0, gp1);
      }
      mark_interior_edge_cuts(k1, k2, gp0, gp1);
      if (k0_size > 1){
        mark_boundary_edge_cuts(k0_size-1, k1, k2, gp0, gp1);
      }
    }
  }
}


template <class Data_Type>
void CSurface<Data_Type>::normals(float *normals){
  int64_t n3 = 3*vertex_count();
  for (int64_t v = 0 ; v < n3 ; v += 3){
    float x[3] = {vxyz.element(v), vxyz.element(v+1), vxyz.element(v+2)};
    float g[3];
    for (int a = 0 ; a < 3 ; ++a){g[a] = (x[a] == 0 ? 1 : (x[a] == size[a]-1 ? -1 : 0));}

    if (g[0] == 0 && g[1] == 0 && g[2] == 0){
      int i[3] = {(int)x[0], (int)x[1], (int)x[2]};
      const Data_Type *ga = grid + stride[0]*(int)i[0] + stride[1]*(int)i[1] + stride[2]*(int)i[2];
      const Data_Type *gb = ga;
      int off[3] = {0,0,0};
      float fb = 0;
      for (int a = 0 ; a < 3 ; ++a){
        if ((fb = x[a]-i[a]) > 0) { off[a] = 1; gb = ga + stride[a]; break; }
      }
      float fa = 1-fb;
      for (int a = 0 ; a < 3 ; ++a){
        int s = stride[a];
        int ia = i[a], ib = ia + off[a];
        // TODO: double check this
        g[a] = (fa*(ia == 0 ? 2*((float)ga[s]-ga[0]) : (float)ga[s]-*(ga-s))
              + fb*(ib == 0 ? 2*((float)gb[s]-gb[0]) :
              ib == size[a]-1 ? 2*((float)gb[0]-*(gb-s))
              : (float)gb[s]-*(gb-s)));
      }
      float norm = sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);

      if (norm > 0)
      {
        g[0] /= norm;
        g[1] /= norm;
        g[2] /= norm;
      }
    }
    normals[v] = -g[0];
    normals[v+1] = -g[1];
    normals[v+2] = -g[2];
  }
}











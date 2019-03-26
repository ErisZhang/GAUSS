#include <GaussIncludes.h>
#include <FEMIncludes.h>
#include <iostream>
 
#include <PhysicalSystemParticles.h>
#include <TimeStepperEulerImplicitLinear.h>
#include <ForceSpring.h>
#include <ConstraintFixedPoint.h>
#include <SolverPardiso.h>

#include <igl/readMESH.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <AssemblerParallel.h>

//solver
#include <igl/active_set.h>


//Global variables for UI
igl::opengl::glfw::Viewer viewer;

SolverPardiso<Eigen::SparseMatrix<double, Eigen::RowMajor>> m_pardiso;

using namespace Gauss;
using namespace Gauss::FEM;

/* Particle Systems */

//Setup the FE System
typedef PhysicalSystemFEM<double, LinearTet> LinearFEM;

typedef World<double, std::tuple<LinearFEM *>,
std::tuple<ForceSpringFEMParticle<double> *>,
std::tuple<ConstraintFixedPoint<double> *> > SimWorld;

//This code from libigl boundary_facets.h
void tetsToTriangles(Eigen::MatrixXi &Fout, Eigen::MatrixXi T) {
    
    unsigned int simplex_size = 4;
    std::vector<std::vector<int> > allF(
                                   T.rows()*simplex_size,
                                        std::vector<int>(simplex_size-1));
    
    // Gather faces, loop over tets
    for(int i = 0; i< (int)T.rows();i++)
    {
        // get face in correct order
        allF[i*simplex_size+0][0] = T(i,2);
        allF[i*simplex_size+0][1] = T(i,3);
        allF[i*simplex_size+0][2] = T(i,1);
        // get face in correct order
        allF[i*simplex_size+1][0] = T(i,3);
        allF[i*simplex_size+1][1] = T(i,2);
        allF[i*simplex_size+1][2] = T(i,0);
        // get face in correct order
        allF[i*simplex_size+2][0] = T(i,1);
        allF[i*simplex_size+2][1] = T(i,3);
        allF[i*simplex_size+2][2] = T(i,0);
        // get face in correct order
        allF[i*simplex_size+3][0] = T(i,2);
        allF[i*simplex_size+3][1] = T(i,1);
        allF[i*simplex_size+3][2] = T(i,0);
    }
    
    Fout.resize(allF.size(), simplex_size-1);
    for(unsigned int ii=0; ii<allF.size(); ++ii) {
        Fout(ii,0) = allF[ii][0];
        Fout(ii,1) = allF[ii][1];
        Fout(ii,2) = allF[ii][2];
    }
}



int main(int argc, char **argv)
{
    std::cout<<"Example 1\n";

    Eigen::MatrixXd V,C;
    Eigen::MatrixXi T,F;
    Eigen::VectorXd s;
    Eigen::VectorXi N;
    
    //Setup shape
    igl::readMESH(dataDir()+"/meshesTetWild/archbridge.mesh", V, T, F); // dataDir using namespace

    tetsToTriangles(F, T);

    //Setup simulation code
    SimWorld world;
    LinearFEM *fem = new LinearFEM(V, T);
    world.addSystem(fem);
    world.finalize();

    //Assemble Stiffness Matrix and get forces
    AssemblerParallel<double, AssemblerEigenSparseMatrix<double>> K;
    AssemblerParallel<double, AssemblerEigenVector<double>> f;
    getStiffnessMatrix(K, world);
    getForceVector(f, world);

    //Project out fixed boundary
    Eigen::VectorXi fixedVertices = minVertices(fem, 1, 1e-3);
    //increase gravity
    Eigen::Vector3d g = {0.f, -100.f, 0.f};
    for(unsigned int ii=0; ii<fem->getImpl().getElements().size(); ++ii) {
        fem->getImpl().getElements()[ii]->setGravity(g);
    }

    Eigen::SparseMatrix<double> P = fixedPointProjectionMatrix(fixedVertices, *fem, world);
    Eigen::SparseMatrix<double> Kp = P*(*K)*P.transpose();
    Eigen::VectorXd fp = P*(*f);

    Eigen::SparseMatrix<double, Eigen::RowMajor> KpR = Kp;
    m_pardiso.symbolicFactorization(KpR);
    m_pardiso.numericalFactorization();
    
    m_pardiso.solve(fp);
    mapStateEigen<0>(world) = P.transpose()*m_pardiso.getX();
    
    // //Solve and project back to the full space
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLT(Kp);
    // mapStateEigen<0>(world) = P.transpose()*LDLT.solve(fp);
    
    //Display norm of stress
    C.resize(V.rows(),3);
    N.resize(V.rows());
    N.setZero();
    Eigen::Matrix3d S;
    s.resize(V.rows());

    for(unsigned int ii=0; ii< T.rows(); ii++) {
        for(unsigned int jj=0; jj<4; ++jj) {
            N[T(ii,jj)] += 1; //get normalization values for each vertex
        }
    }

    for(unsigned int ii=0; ii< T.rows(); ii++) {
        for(unsigned int jj=0; jj<4; ++jj) {
            fem->getImpl().getElement(ii)->getCauchyStress(S, Vec3d(0,0,0), world.getState());
            s(T(ii,jj)) += S.norm()/static_cast<double>(N(T(ii,jj)));
        }
    }

    igl::jet(s, 0, 100000, C);
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.launch();

    return 0;
}


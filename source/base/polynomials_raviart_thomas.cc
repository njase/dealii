// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


#include <deal.II/base/utilities.h>
#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>
#include <iostream>
#include <iomanip>

//TODO[WB]: This class is not thread-safe: it uses mutable member variables that contain temporary state. this is not what one would want when one uses a finite element object in a number of different contexts on different threads: finite element objects should be stateless
//TODO:[GK] This can be achieved by writing a function in Polynomial space which does the rotated fill performed below and writes the data into the right data structures. The same function would be used
//by Nedelec polynomials.

DEAL_II_NAMESPACE_OPEN


template <int dim>
PolynomialsRaviartThomas<dim>::PolynomialsRaviartThomas (const unsigned int k)
  :
  my_degree(k),
  polynomial_space (create_polynomials (k)),
  n_pols(compute_n_pols(k))
{
	create_poly_mapping();
}



template <int dim>
std::vector<std::vector< Polynomials::Polynomial< double > > >
PolynomialsRaviartThomas<dim>::create_polynomials (const unsigned int k)
{
  std::vector<std::vector< Polynomials::Polynomial< double > > > pols(dim);
  pols[0] = Polynomials::LagrangeEquidistant::generate_complete_basis(k+1);
  if (k == 0)
    for (unsigned int d=1; d<dim; ++d)
      pols[d] = Polynomials::Legendre::generate_complete_basis(0);
  else
    for (unsigned int d=1; d<dim; ++d)
      pols[d] = Polynomials::LagrangeEquidistant::generate_complete_basis(k);

  return pols;
}


template <int dim>
void
PolynomialsRaviartThomas<dim>::compute (const Point<dim>            &unit_point,
                                        std::vector<Tensor<1,dim> > &values,
                                        std::vector<Tensor<2,dim> > &grads,
                                        std::vector<Tensor<3, dim> > &grad_grads,
                                        std::vector<Tensor<4, dim> > &third_derivatives,
                                        std::vector<Tensor<5, dim> > &fourth_derivatives) const
{
  Assert(values.size()==n_pols || values.size()==0,
         ExcDimensionMismatch(values.size(), n_pols));
  Assert(grads.size()==n_pols|| grads.size()==0,
         ExcDimensionMismatch(grads.size(), n_pols));
  Assert(grad_grads.size()==n_pols|| grad_grads.size()==0,
         ExcDimensionMismatch(grad_grads.size(), n_pols));
  Assert(third_derivatives.size()==n_pols|| third_derivatives.size()==0,
         ExcDimensionMismatch(third_derivatives.size(), n_pols));
  Assert(fourth_derivatives.size()==n_pols|| fourth_derivatives.size()==0,
         ExcDimensionMismatch(fourth_derivatives.size(), n_pols));

  // have a few scratch
  // arrays. because we don't want to
  // re-allocate them every time this
  // function is called, we make them
  // static. however, in return we
  // have to ensure that the calls to
  // the use of these variables is
  // locked with a mutex. if the
  // mutex is removed, several tests
  // (notably
  // deal.II/create_mass_matrix_05)
  // will start to produce random
  // results in multithread mode
  static Threads::Mutex mutex;
  Threads::Mutex::ScopedLock lock(mutex);

  static std::vector<double> p_values;
  static std::vector<Tensor<1,dim> > p_grads;
  static std::vector<Tensor<2,dim> > p_grad_grads;
  static std::vector<Tensor<3,dim> > p_third_derivatives;
  static std::vector<Tensor<4,dim> > p_fourth_derivatives;

  const unsigned int n_sub = polynomial_space.n();
  p_values.resize((values.size() == 0) ? 0 : n_sub);
  p_grads.resize((grads.size() == 0) ? 0 : n_sub);
  p_grad_grads.resize((grad_grads.size() == 0) ? 0 : n_sub);
  p_third_derivatives.resize((third_derivatives.size() == 0) ? 0 : n_sub);
  p_fourth_derivatives.resize((fourth_derivatives.size() == 0) ? 0 : n_sub);

	std::cout<<"tensor_pols_grad_mapping_inv[0][0]"<<std::endl;
	for (int i=0; i<n_sub;i++)
	{
		std::cout<<tensor_pols_grad_mapping_inv[0][0][i]<<"    "<<std::endl;
	}
	std::cout<<std::endl;

	std::cout<<"tensor_pols_grad_mapping_inv[0][1]"<<std::endl;
	for (int i=0; i<n_sub;i++)
	{
		std::cout<<tensor_pols_grad_mapping_inv[0][1][i]<<"    "<<std::endl;
	}
		std::cout<<std::endl;

  for (unsigned int d=0; d<dim; ++d)
    {
      // First we copy the point. The
      // polynomial space for
      // component d consists of
      // polynomials of degree k+1 in
      // x_d and degree k in the
      // other variables. in order to
      // simplify this, we use the
      // same AnisotropicPolynomial
      // space and simply rotate the
      // coordinates through all
      // directions.
      Point<dim> p;
      for (unsigned int c=0; c<dim; ++c)
        p(c) = unit_point((c+d)%dim);

      polynomial_space.compute (p, p_values, p_grads, p_grad_grads, p_third_derivatives, p_fourth_derivatives);

      for (unsigned int i=0; i<p_values.size(); ++i)
        values[i+d*n_sub][d] = p_values[tensor_pols_mapping_inv[d][i]];

      for (unsigned int i=0; i<p_grads.size(); ++i)
        for (unsigned int d1=0; d1<dim; ++d1)
        {
        	if (d==0)
        	{
        		std::cout<<"Direction = "<<(d1+d)%dim<<" Filling from index "<<tensor_pols_grad_mapping_inv[d][(d1+d)%dim][i]<<std::endl;
        	}
        	grads[i+d*n_sub][d][(d1+d)%dim] = p_grads[tensor_pols_grad_mapping_inv[d][(d1+d)%dim][i]][d1];
        }

      for (unsigned int i=0; i<p_grad_grads.size(); ++i)
        for (unsigned int d1=0; d1<dim; ++d1)
          for (unsigned int d2=0; d2<dim; ++d2)
            grad_grads[i+d*n_sub][d][(d1+d)%dim][(d2+d)%dim]
              = p_grad_grads[tensor_pols_mapping_inv[d][i]][d1][d2];

      for (unsigned int i=0; i<p_third_derivatives.size(); ++i)
        for (unsigned int d1=0; d1<dim; ++d1)
          for (unsigned int d2=0; d2<dim; ++d2)
            for (unsigned int d3=0; d3<dim; ++d3)
              third_derivatives[i+d*n_sub][d][(d1+d)%dim][(d2+d)%dim][(d3+d)%dim]
                = p_third_derivatives[tensor_pols_mapping_inv[d][i]][d1][d2][d3];

      for (unsigned int i=0; i<p_fourth_derivatives.size(); ++i)
        for (unsigned int d1=0; d1<dim; ++d1)
          for (unsigned int d2=0; d2<dim; ++d2)
            for (unsigned int d3=0; d3<dim; ++d3)
              for (unsigned int d4=0; d4<dim; ++d4)
                fourth_derivatives[i+d*n_sub][d][(d1+d)%dim][(d2+d)%dim][(d3+d)%dim][(d4+d)%dim]
                  = p_fourth_derivatives[tensor_pols_mapping_inv[d][i]][d1][d2][d3][d4];
    }
}


template <int dim>
unsigned int
PolynomialsRaviartThomas<dim>::compute_n_pols(unsigned int k)
{
  if (dim == 1) return k+1;
  if (dim == 2) return 2*(k+1)*(k+2);
  if (dim == 3) return 3*(k+1)*(k+1)*(k+2);

  Assert(false, ExcNotImplemented());
  return 0;
}

template <int dim>
void
PolynomialsRaviartThomas<dim>::create_poly_mapping ()
{
	if ((dim > 2)) //FIXME: For dim=3
		Assert(false, ExcNotImplemented());

	constexpr int n_components = dim; //For clarity

	const unsigned int n_sub = polynomial_space.n();

	for (int c=0;c<n_components;c++)
	{
		tensor_pols_mapping_inv[c].resize(n_sub);
		for (int d=0; d<dim; d++)
			tensor_pols_grad_mapping_inv[c][d].resize(n_sub);
	}

	////Fill for values
	//For first component, mapping is 1:1
	for (int i=0; i<n_sub; i++)
	{
		tensor_pols_mapping_inv[0][i] = i;
	}

	tensor_pols_grad_mapping_inv[0][0] = tensor_pols_mapping_inv[0];
	tensor_pols_grad_mapping_inv[0][1] = tensor_pols_mapping_inv[0];

	//FIXME This is only for dim=2
	//Actual indices are locations corresponding to transposed matrix of calculated indices
	const unsigned int ny = my_degree+2, nx=my_degree+1; //no of dofs in each direction
	if ((nx*ny) != n_sub)
		Assert(false, ExcInternalError());

	int k=0, k_act=0;
	//Store the indices cooresponding to matrix transpose
	for (int i=0; i<ny; i++)
	{
		for (int j=0; j<nx; j++)
		{
			k_act = i*nx+j;
			k = j*ny+i;
			tensor_pols_mapping_inv[1][k_act] = k;
		}
	}

	//Fill for derivatives - same for all directions  FIXME: This is currently only for dim=2
	tensor_pols_grad_mapping_inv[1][0] = tensor_pols_mapping_inv[1];
	tensor_pols_grad_mapping_inv[1][1] = tensor_pols_mapping_inv[1];
}

template class PolynomialsRaviartThomas<1>;
template class PolynomialsRaviartThomas<2>;
template class PolynomialsRaviartThomas<3>;


DEAL_II_NAMESPACE_CLOSE

// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2017 by the deal.II authors
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

#ifndef dealii_matrix_free_shape_info_templates_h
#define dealii_matrix_free_shape_info_templates_h


#include <deal.II/base/utilities.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/base/aligned_vector.h> //Debug only

#include <deal.II/matrix_free/shape_info.h>


DEAL_II_NAMESPACE_OPEN


namespace internal
{
  namespace MatrixFreeFunctions
  {
    // ----------------- actual ShapeInfo functions --------------------

    namespace
    {
      template <typename Number>
      Number get_first_array_element(const Number a)
      {
        return a;
      }

      template <typename Number>
      Number get_first_array_element(const VectorizedArray<Number> a)
      {
        return a[0];
      }
    }

    template <typename Number>
    ShapeInfo<Number>::ShapeInfo ()
      :
      element_type(tensor_general),
      use_non_primitive(false),
      fe_degree(numbers::invalid_unsigned_int),
      n_q_points_1d(0),
      n_q_points (0),
      dofs_per_component_on_cell (0),
      n_q_points_face (0),
      dofs_per_component_on_face (0),
      nodal_at_cell_boundaries (false)
    {}



    template <typename Number>
    template <int dim>
    void
    ShapeInfo<Number>::reinit (const Quadrature<1> &quad,
                               const FiniteElement<dim> &fe_in,
                               const unsigned int base_element_number
                               )
    {
    	//Due to structure of FE which we use (RT and FE_Q), we want to store data for max two dimensions only
    	base_shape_values.resize(2);
    	base_shape_gradients.resize(2);
    	base_shape_hessians.resize(2);

    	if (dynamic_cast<const FE_RaviartThomas<dim> *>(&fe_in))
    		use_non_primitive = true;
    	else if (fe_in.n_components() == dim)
    		use_non_primitive = true; //This is only for R&D and experimentation

    	if (use_non_primitive && (fe_in.n_components() == dim))
    		internal_reinit_vector(quad,fe_in,base_element_number);
    	else
    		internal_reinit_scalar(quad,fe_in,base_element_number);
    }



    template <typename Number>
    bool
    ShapeInfo<Number>::check_1d_shapes_symmetric(const unsigned int n_q_points_1d)
    {
      if (dofs_per_component_on_cell == 0)
        return false;

      const double zero_tol =
        std::is_same<Number,double>::value==true?1e-12:1e-7;
      // symmetry for values
      const unsigned int n_dofs_1d = fe_degree + 1;
      for (unsigned int i=0; i<(n_dofs_1d+1)/2; ++i)
        for (unsigned int j=0; j<n_q_points_1d; ++j)
          if (std::abs(get_first_array_element(shape_values[i*n_q_points_1d+j] -
                                               shape_values[(n_dofs_1d-i)*n_q_points_1d-j-1])) >
              std::max(zero_tol, zero_tol*
                       std::abs(get_first_array_element(shape_values[i*n_q_points_1d+j]))))
            return false;

      // shape values should be zero at x=0.5 for all basis functions except
      // for one which is one
      if (n_q_points_1d%2 == 1 && n_dofs_1d%2 == 1)
        {
          for (unsigned int i=0; i<n_dofs_1d/2; ++i)
            if (std::abs(get_first_array_element(shape_values[i*n_q_points_1d+
                                                              n_q_points_1d/2])) > zero_tol)
              return false;
          if (std::abs(get_first_array_element(shape_values[(n_dofs_1d/2)*n_q_points_1d+
                                                            n_q_points_1d/2])-1.)> zero_tol)
            return false;
        }

      // skew-symmetry for gradient, zero of middle basis function in middle
      // quadrature point. Multiply tolerance by degree of the element to
      // the power of 1.5 to get a suitable gradient scaling
      const double zero_tol_gradient = zero_tol * std::sqrt(fe_degree+1.)*(fe_degree+1);
      for (unsigned int i=0; i<(n_dofs_1d+1)/2; ++i)
        for (unsigned int j=0; j<n_q_points_1d; ++j)
          if (std::abs(get_first_array_element(shape_gradients[i*n_q_points_1d+j] +
                                               shape_gradients[(n_dofs_1d-i)*n_q_points_1d-
                                                               j-1])) > zero_tol_gradient)
            return false;
      if (n_dofs_1d%2 == 1 && n_q_points_1d%2 == 1)
        if (std::abs(get_first_array_element(shape_gradients[(n_dofs_1d/2)*n_q_points_1d+
                                                             (n_q_points_1d/2)]))
            > zero_tol_gradient)
          return false;

      // symmetry for Hessian. Multiply tolerance by degree^3 of the element
      // to get a suitable Hessian scaling
      const double zero_tol_hessian = zero_tol * (fe_degree+1)*(fe_degree+1)*(fe_degree+1);
      for (unsigned int i=0; i<(n_dofs_1d+1)/2; ++i)
        for (unsigned int j=0; j<n_q_points_1d; ++j)
          if (std::abs(get_first_array_element(shape_hessians[i*n_q_points_1d+j] -
                                               shape_hessians[(n_dofs_1d-i)*n_q_points_1d-
                                                              j-1])) > zero_tol_hessian)
            return false;

      const unsigned int stride = (n_q_points_1d+1)/2;
      shape_values_eo.resize((fe_degree+1)*stride);
      shape_gradients_eo.resize((fe_degree+1)*stride);
      shape_hessians_eo.resize((fe_degree+1)*stride);
      for (unsigned int i=0; i<(fe_degree+1)/2; ++i)
        for (unsigned int q=0; q<stride; ++q)
          {
            shape_values_eo[i*stride+q] =
              0.5 * (shape_values[i*n_q_points_1d+q] +
                     shape_values[i*n_q_points_1d+n_q_points_1d-1-q]);
            shape_values_eo[(fe_degree-i)*stride+q] =
              0.5 * (shape_values[i*n_q_points_1d+q] -
                     shape_values[i*n_q_points_1d+n_q_points_1d-1-q]);

            shape_gradients_eo[i*stride+q] =
              0.5 * (shape_gradients[i*n_q_points_1d+q] +
                     shape_gradients[i*n_q_points_1d+n_q_points_1d-1-q]);
            shape_gradients_eo[(fe_degree-i)*stride+q] =
              0.5 * (shape_gradients[i*n_q_points_1d+q] -
                     shape_gradients[i*n_q_points_1d+n_q_points_1d-1-q]);

            shape_hessians_eo[i*stride+q] =
              0.5 * (shape_hessians[i*n_q_points_1d+q] +
                     shape_hessians[i*n_q_points_1d+n_q_points_1d-1-q]);
            shape_hessians_eo[(fe_degree-i)*stride+q] =
              0.5 * (shape_hessians[i*n_q_points_1d+q] -
                     shape_hessians[i*n_q_points_1d+n_q_points_1d-1-q]);
          }
      if (fe_degree % 2 == 0)
        for (unsigned int q=0; q<stride; ++q)
          {
            shape_values_eo[fe_degree/2*stride+q] =
              shape_values[(fe_degree/2)*n_q_points_1d+q];
            shape_gradients_eo[fe_degree/2*stride+q] =
              shape_gradients[(fe_degree/2)*n_q_points_1d+q];
            shape_hessians_eo[fe_degree/2*stride+q] =
              shape_hessians[(fe_degree/2)*n_q_points_1d+q];
          }

      return true;
    }



    template <typename Number>
    bool
    ShapeInfo<Number>::check_1d_shapes_collocation()
    {
      if (dofs_per_component_on_cell != n_q_points)
        return false;

      const double zero_tol =
        std::is_same<Number,double>::value==true?1e-12:1e-7;
      // check: identity operation for shape values
      const unsigned int n_points_1d = fe_degree+1;
      for (unsigned int i=0; i<n_points_1d; ++i)
        for (unsigned int j=0; j<n_points_1d; ++j)
          if (i!=j)
            {
              if (std::abs(get_first_array_element(shape_values[i*n_points_1d+j]))>zero_tol)
                return false;
            }
          else
            {
              if (std::abs(get_first_array_element(shape_values[i*n_points_1d+j])-1.)>zero_tol)
                return false;
            }
      return true;
    }



    template <typename Number>
    std::size_t
    ShapeInfo<Number>::memory_consumption () const
    {
      std::size_t memory = sizeof(*this);
      memory += MemoryConsumption::memory_consumption(shape_values);
      memory += MemoryConsumption::memory_consumption(shape_gradients);
      memory += MemoryConsumption::memory_consumption(shape_hessians);
      memory += MemoryConsumption::memory_consumption(shape_values_eo);
      memory += MemoryConsumption::memory_consumption(shape_gradients_eo);
      memory += MemoryConsumption::memory_consumption(shape_hessians_eo);
      memory += MemoryConsumption::memory_consumption(shape_gradients_collocation_eo);
      memory += MemoryConsumption::memory_consumption(shape_hessians_collocation_eo);
      for (unsigned int i=0; i<2; ++i)
        {
          memory += MemoryConsumption::memory_consumption(shape_data_on_face[i]);
          memory += MemoryConsumption::memory_consumption(values_within_subface[i]);
          memory += MemoryConsumption::memory_consumption(gradients_within_subface[i]);
        }
      return memory;
    }

    template <typename Number>
    template <int dim>
    std::vector<unsigned int>
    ShapeInfo<Number>::lexicographic_renumber(const FiniteElement<dim> &fe_in, const unsigned int base_element_number)
    {
    	const FiniteElement<dim> *fe = &fe_in.base_element(base_element_number);

        // renumber (this is necessary for FE_Q, for example, since there the
        // vertex DoFs come first, which is incompatible with the lexicographic
        // ordering necessary to apply tensor products efficiently)
        std::vector<unsigned int> scalar_lexicographic;
        {
          const FE_Poly<TensorProductPolynomials<dim>,dim,dim> *fe_poly =
            dynamic_cast<const FE_Poly<TensorProductPolynomials<dim>,dim,dim>*>(fe);

          const FE_Poly<TensorProductPolynomials<dim,Polynomials::
          PiecewisePolynomial<double> >,dim,dim> *fe_poly_piece =
            dynamic_cast<const FE_Poly<TensorProductPolynomials<dim,
            Polynomials::PiecewisePolynomial<double> >,dim,dim>*> (fe);

          const FE_DGP<dim> *fe_dgp = dynamic_cast<const FE_DGP<dim>*>(fe);

          const FE_Q_DG0<dim> *fe_q_dg0 = dynamic_cast<const FE_Q_DG0<dim>*>(fe);

          element_type = tensor_general;
          if (fe_poly != nullptr)
            scalar_lexicographic = fe_poly->get_poly_space_numbering_inverse();
          else if (fe_poly_piece != nullptr)
            scalar_lexicographic = fe_poly_piece->get_poly_space_numbering_inverse();
          else if (fe_dgp != nullptr)
            {
              scalar_lexicographic.resize(fe_dgp->dofs_per_cell);
              for (unsigned int i=0; i<fe_dgp->dofs_per_cell; ++i)
                scalar_lexicographic[i] = i;
              element_type = truncated_tensor;
            }
          else if (fe_q_dg0 != nullptr)
            {
              scalar_lexicographic = fe_q_dg0->get_poly_space_numbering_inverse();
              element_type = tensor_symmetric_plus_dg0;
            }
          else if (fe->dofs_per_cell == 0)
            {
              // FE_Nothing case -> nothing to do here
            }
          else
            Assert(false, ExcNotImplemented());

          // Finally store the renumbering into the member variable of this
          // class
          if (fe_in.n_components() == 1)
            lexicographic_numbering = scalar_lexicographic;
          else
            {
              // have more than one component, get the inverse
              // permutation, invert it, sort the components one after one,
              // and invert back
              std::vector<unsigned int> scalar_inv =
                Utilities::invert_permutation(scalar_lexicographic);
              std::vector<unsigned int> lexicographic(fe_in.dofs_per_cell,
                                                      numbers::invalid_unsigned_int);
              const unsigned int mul = fe_in.element_multiplicity(base_element_number);
              unsigned int components_before = 0;
              for (unsigned int e=0; e<base_element_number; ++e)
                components_before += fe_in.element_multiplicity(e);
              for (unsigned int comp=0;comp<mul; ++comp)
                for (unsigned int i=0; i<scalar_inv.size(); ++i)
                	lexicographic[i*mul+comp+components_before]
                    = scalar_inv.size () * comp + scalar_inv[i];  //TBD: Bug yet to be reported in dealii

              // invert numbering again. Need to do it manually because we might
              // have undefined blocks
              lexicographic_numbering.resize(fe_in.element_multiplicity(base_element_number)*fe->dofs_per_cell, numbers::invalid_unsigned_int);
              for (unsigned int i=0; i<lexicographic.size(); ++i)
                if (lexicographic[i] != numbers::invalid_unsigned_int)
                  {
                    AssertIndexRange(lexicographic[i],
                                     lexicographic_numbering.size());
                    lexicographic_numbering[lexicographic[i]] = i;
                  }
            }
        }

        return scalar_lexicographic;
    }

    //FIXME : Currently this supports dim=2 only, and k > 2,3.
    //Old ordering:
    //			 <face points for component 1, face points for compnent 2...>,
    //			 <1st interior point for component 1, 1st interior points for component 2...>
    //			 <2nd interior point for component 1, 2nd interior points for component 2...>
    //			 .....
    //New ordering:
    //			<Face points, interior points for component 1>,
    //			<Face points, interior points for component 2>,
    //           ....and so on
    //  The new ordering shows TP structure in each component
    template <typename Number>
    template <int dim>
    void
    ShapeInfo<Number>::raviart_thomas_lexicographic_renumber(const FE_RaviartThomas<dim> * fe)
    {
    	lexicographic_numbering.resize(fe->dofs_per_cell, numbers::invalid_unsigned_int);

    	const int nfp_per_comp = fe->n_face_dofs/dim;
    	const int nip_per_comp = fe->n_interior_dofs/dim;
    	const int n_per_comp = nfp_per_comp + nip_per_comp;

    	//Step1 : Reorder such that all (face+interior) points of a component are together, in the order of appearance
    	int current = 0;
    	for (int d=0; d<dim; d++)
    		for (int i=0; i<nfp_per_comp; i++)
    			lexicographic_numbering[n_per_comp*d+i] = current++;

    	for (int i=0; i<nip_per_comp; i++)
    		for (int d=0; d<dim; d++)
    			lexicographic_numbering[n_per_comp*d+nfp_per_comp+i] = current++;

#ifdef __UT__
    	//Only for debugging - uptil here is fine for any degree in dim2
    	std::cout<<std::endl;
		std::cout<<"First reordering lexicographic result is"<<std::endl;
		for (unsigned int j=0; j<lexicographic_numbering.size(); j++)
		{
			std::cout <<"   "<<lexicographic_numbering[j];
		}
#endif


    	//Step2: Reorder points of first component so that result has tensor product structure. second component already
    	//       shows tensor product structure
    	std::vector<unsigned int> temp(lexicographic_numbering);
    	const int int_values_per_iter = fe->degree-1;
    	const int face_values_per_iter = ((nfp_per_comp/nip_per_comp) > int_values_per_iter) ?
    															(nfp_per_comp/nip_per_comp) :
    															int_values_per_iter;

    	for (int n = 0; n<n_per_comp/(face_values_per_iter+int_values_per_iter); n++)
    	{
    		current = n*(face_values_per_iter+int_values_per_iter);
    		for (int i=0; i<face_values_per_iter; i++)
    			lexicographic_numbering[current+i] = temp[n+i*(int_values_per_iter+1)]; //for face points

    		current += face_values_per_iter;
    		for (int i=0; i<int_values_per_iter; i++)
    			lexicographic_numbering[current+i] = temp[nfp_per_comp+n*int_values_per_iter+i]; //for internal points
    	}

#ifdef __UT__
    	//Only for debugging - incorrect result for deg>2
    	std::cout<<std::endl;
		std::cout<<"lexicographic result is"<<std::endl;
		for (unsigned int j=0; j<lexicographic_numbering.size(); j++)
		{
			std::cout <<"   "<<lexicographic_numbering[j];
			//std::cout<<std::endl;
		}
#endif
    }


    template <typename Number>
    template <int dim>
    void ShapeInfo<Number>::internal_reinit_scalar (const Quadrature<1> &quad,
                 const FiniteElement<dim> &fe_in,
                 const unsigned int base_element_number)
    {
        const FiniteElement<dim> *fe = &fe_in.base_element(base_element_number);

        Assert (fe->n_components() == 1,
                ExcMessage("FEEvaluation only works for scalar finite elements."));

        fe_degree = fe->degree;
        n_q_points_1d = quad.size();

        const unsigned int n_dofs_1d = std::min(fe->dofs_per_cell, fe_degree+1);

        // renumber (this is necessary for FE_Q, for example, since there the
        // vertex DoFs come first, which is incompatible with the lexicographic
        // ordering necessary to apply tensor products efficiently)
        std::vector<unsigned int> scalar_lexicographic;
        Point<dim> unit_point;

        // find numbering to lexicographic
        Assert(fe->n_components() == 1,
               ExcMessage("Expected a scalar element"));

        scalar_lexicographic = lexicographic_renumber(fe_in,base_element_number);
        // to evaluate 1D polynomials, evaluate along the line with the first
        // unit support point, assuming that fe.shape_value(0,unit_point) ==
        // 1. otherwise, need other entry point (e.g. generating a 1D element
        // by reading the name, as done before r29356)
        if (fe->has_support_points())
          unit_point = fe->get_unit_support_points()[scalar_lexicographic[0]];
        Assert(fe->dofs_per_cell == 0 ||
               std::abs(fe->shape_value(scalar_lexicographic[0],
                                        unit_point)-1) < 1e-13,
               ExcInternalError("Could not decode 1D shape functions for the "
                                "element " + fe->get_name()));


        n_q_points      = Utilities::fixed_power<dim>(n_q_points_1d);
        n_q_points_face = dim>1?Utilities::fixed_power<dim-1>(n_q_points_1d):1;
        dofs_per_component_on_cell = fe->dofs_per_cell;
        dofs_per_component_on_face = dim>1?Utilities::fixed_power<dim-1>(fe_degree+1):1;

        const unsigned int array_size = n_dofs_1d*n_q_points_1d;

        this->shape_gradients.resize_fast (array_size);
        this->shape_values.resize_fast (array_size);
        this->shape_hessians.resize_fast (array_size);

        this->shape_data_on_face[0].resize(3*n_dofs_1d);
        this->shape_data_on_face[1].resize(3*n_dofs_1d);
        this->values_within_subface[0].resize(array_size);
        this->values_within_subface[1].resize(array_size);
        this->gradients_within_subface[0].resize(array_size);
        this->gradients_within_subface[1].resize(array_size);
        this->hessians_within_subface[0].resize(array_size);
        this->hessians_within_subface[1].resize(array_size);

        for (unsigned int i=0; i<n_dofs_1d; ++i)
          {
            // need to reorder from hierarchical to lexicographic to get the
            // DoFs correct
            const unsigned int my_i = scalar_lexicographic[i];
            for (unsigned int q=0; q<n_q_points_1d; ++q)
              {
                Point<dim> q_point = unit_point;
                q_point[0] = quad.get_points()[q][0];

                shape_values   [i*n_q_points_1d+q] = fe->shape_value(my_i,q_point);
                shape_gradients[i*n_q_points_1d+q] = fe->shape_grad(my_i,q_point)[0];
                shape_hessians [i*n_q_points_1d+q] = fe->shape_grad_grad(my_i,q_point)[0][0];

                // evaluate basis functions on the two 1D subfaces (i.e., at the
                // positions divided by one half and shifted by one half,
                // respectively)
                q_point[0] *= 0.5;
                values_within_subface[0][i*n_q_points_1d+q] = fe->shape_value(my_i,q_point);
                gradients_within_subface[0][i*n_q_points_1d+q] = fe->shape_grad(my_i,q_point)[0];
                hessians_within_subface[0][i*n_q_points_1d+q] = fe->shape_grad_grad(my_i,q_point)[0][0];
                q_point[0] += 0.5;
                values_within_subface[1][i*n_q_points_1d+q] = fe->shape_value(my_i,q_point);
                gradients_within_subface[1][i*n_q_points_1d+q] = fe->shape_grad(my_i,q_point)[0];
                hessians_within_subface[1][i*n_q_points_1d+q] = fe->shape_grad_grad(my_i,q_point)[0][0];
              }

            // evaluate basis functions on the 1D faces, i.e., in zero and one
            Point<dim> q_point = unit_point;
            q_point[0] = 0;
            this->shape_data_on_face[0][i] = fe->shape_value(my_i,q_point);
            this->shape_data_on_face[0][i+n_dofs_1d] = fe->shape_grad(my_i,q_point)[0];
            this->shape_data_on_face[0][i+2*n_dofs_1d] = fe->shape_grad_grad(my_i,q_point)[0][0];
            q_point[0] = 1;
            this->shape_data_on_face[1][i] = fe->shape_value(my_i,q_point);
            this->shape_data_on_face[1][i+n_dofs_1d] = fe->shape_grad(my_i,q_point)[0];
            this->shape_data_on_face[1][i+2*n_dofs_1d] = fe->shape_grad_grad(my_i,q_point)[0][0];
          }

        // get gradient and Hessian transformation matrix for the polynomial
        // space associated with the quadrature rule (collocation space)
        {
          const unsigned int stride = (n_q_points_1d+1)/2;
          shape_gradients_collocation_eo.resize(n_q_points_1d*stride);
          shape_hessians_collocation_eo.resize(n_q_points_1d*stride);
          FE_DGQArbitraryNodes<1> fe(quad.get_points());
          for (unsigned int i=0; i<n_q_points_1d/2; ++i)
            for (unsigned int q=0; q<stride; ++q)
              {
                shape_gradients_collocation_eo[i*stride+q] =
                  0.5* (fe.shape_grad(i, quad.get_points()[q])[0] +
                        fe.shape_grad(i, quad.get_points()[n_q_points_1d-1-q])[0]);
                shape_gradients_collocation_eo[(n_q_points_1d-1-i)*stride+q] =
                  0.5* (fe.shape_grad(i, quad.get_points()[q])[0] -
                        fe.shape_grad(i, quad.get_points()[n_q_points_1d-1-q])[0]);
                shape_hessians_collocation_eo[i*stride+q] =
                  0.5* (fe.shape_grad_grad(i, quad.get_points()[q])[0][0] +
                        fe.shape_grad_grad(i, quad.get_points()[n_q_points_1d-1-q])[0][0]);
                shape_hessians_collocation_eo[(n_q_points_1d-1-i)*stride+q] =
                  0.5* (fe.shape_grad_grad(i, quad.get_points()[q])[0][0] -
                        fe.shape_grad_grad(i, quad.get_points()[n_q_points_1d-1-q])[0][0]);
              }
          if (n_q_points_1d % 2 == 1)
            for (unsigned int q=0; q<stride; ++q)
              {
                shape_gradients_collocation_eo[n_q_points_1d/2*stride+q] =
                  fe.shape_grad(n_q_points_1d/2, quad.get_points()[q])[0];
                shape_hessians_collocation_eo[n_q_points_1d/2*stride+q] =
                  fe.shape_grad_grad(n_q_points_1d/2, quad.get_points()[q])[0][0];
              }
        }

        if (element_type == tensor_general &&
            check_1d_shapes_symmetric(n_q_points_1d))
          {
            if (check_1d_shapes_collocation())
              element_type = tensor_symmetric_collocation;
            else
              element_type = tensor_symmetric;
            if (n_dofs_1d > 3 && element_type == tensor_symmetric)
              {
                // check if we are a Hermite type
                element_type = tensor_symmetric_hermite;
                for (unsigned int i=1; i<n_dofs_1d; ++i)
                  if (std::abs(get_first_array_element(shape_data_on_face[0][i])) > 1e-12)
                    element_type = tensor_symmetric;
                for (unsigned int i=2; i<n_dofs_1d; ++i)
                  if (std::abs(get_first_array_element(shape_data_on_face[0][n_dofs_1d+i])) > 1e-12)
                    element_type = tensor_symmetric;
              }
          }
        else if (element_type == tensor_symmetric_plus_dg0)
          check_1d_shapes_symmetric(n_q_points_1d);

        nodal_at_cell_boundaries = true;
        for (unsigned int i=1; i<n_dofs_1d; ++i)
          if (std::abs(get_first_array_element(shape_data_on_face[0][i])) > 1e-13 ||
              std::abs(get_first_array_element(shape_data_on_face[1][i-1])) > 1e-13)
            nodal_at_cell_boundaries = false;

        if (nodal_at_cell_boundaries == true)
          {
            face_to_cell_index_nodal.reinit(GeometryInfo<dim>::faces_per_cell,
                                            dofs_per_component_on_face);
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              {
                const unsigned int direction = f/2;
                const unsigned int stride = direction < dim-1 ? (fe_degree+1) : 1;
                int shift = 1;
                for (unsigned int d=0; d<direction; ++d)
                  shift *= fe_degree+1;
                const unsigned int offset = (f%2)*fe_degree*shift;

                if (direction == 0 || direction == dim-1)
                  for (unsigned int i=0; i<dofs_per_component_on_face; ++i)
                    face_to_cell_index_nodal(f,i) = offset + i*stride;
                else
                  // local coordinate system on faces 2 and 3 is zx in
                  // deal.II, not xz as expected for tensor products -> adjust
                  // that here
                  for (unsigned int j=0; j<=fe_degree; ++j)
                    for (unsigned int i=0; i<=fe_degree; ++i)
                      {
                        const unsigned int ind = offset + j*dofs_per_component_on_face + i;
                        AssertIndexRange(ind, dofs_per_component_on_cell);
                        const unsigned int l = i*(fe_degree+1)+j;
                        face_to_cell_index_nodal(f,l) = ind;
                      }
              }
          }

        if (element_type == tensor_symmetric_hermite)
          {
            face_to_cell_index_hermite.reinit(GeometryInfo<dim>::faces_per_cell,
                                              2*dofs_per_component_on_face);
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              {
                const unsigned int direction = f/2;
                const unsigned int stride = direction < dim-1 ? (fe_degree+1) : 1;
                int shift = 1;
                for (unsigned int d=0; d<direction; ++d)
                  shift *= fe_degree+1;
                const unsigned int offset = (f%2)*fe_degree*shift;
                if (f%2 == 1)
                  shift = -shift;

                if (direction == 0 || direction == dim-1)
                  for (unsigned int i=0; i<dofs_per_component_on_face; ++i)
                    {
                      face_to_cell_index_hermite(f,2*i) = offset + i*stride;
                      face_to_cell_index_hermite(f,2*i+1) = offset + i*stride + shift;
                    }
                else
                  // local coordinate system on faces 2 and 3 is zx in
                  // deal.II, not xz as expected for tensor products -> adjust
                  // that here
                  for (unsigned int j=0; j<=fe_degree; ++j)
                    for (unsigned int i=0; i<=fe_degree; ++i)
                      {
                        const unsigned int ind = offset + j*dofs_per_component_on_face + i;
                        AssertIndexRange(ind, dofs_per_component_on_cell);
                        const unsigned int l = i*(fe_degree+1)+j;
                        face_to_cell_index_hermite(f,2*l) = ind;
                        face_to_cell_index_hermite(f,2*l+1) = ind+shift;
                      }
              }
          }
    }


    template <typename Number>
    template <int dim>
    void ShapeInfo<Number>::internal_reinit_vector (const Quadrature<1> &quad,
                 const FiniteElement<dim> &fe_in,
                 const unsigned int base_element_number)
    {
    	unsigned int vector_n_components = fe_in.n_components();
    	//FIXME Extend for dim=3
    	if (vector_n_components > 2 || dim > 2)
    		Assert (false, ExcNotImplemented());


    	enum class FEName { FE_Unknown=0, FE_RT=1, FE_Q_TP=2 };
    	FEName fe_name = FEName::FE_Unknown;

    	typedef std::pair<unsigned int, unsigned int> mindex; /*comp, dim */
    	std::map<mindex, unsigned int> index_map;

    	unsigned int base_values_count = 0;
    	std::array<unsigned int, 2> n_dofs_1d; //not more than 2 distinct values in case of RT

        /*
         * Algo
         * Using FE Type, identify number of components
         * utilize the reinit logic of ShapeInfo (restructure it) to evaluate values, quad and hessians
         *  for the required 1-D quad points and 1-D basis functions as many times as required for the particular
         *  FEType. Store the results in basic_shape_values
         * perform reinit for all the components as:
    	 *   Mix and match results from basic_shape_values to shape_values_component as needed
         */

    	const FiniteElement<dim> *fe = &fe_in.base_element(base_element_number);

    	if (dynamic_cast<const FE_RaviartThomas<dim> *>(fe))
    	{
        	if (dim != vector_n_components)
        		Assert (false, ExcNotImplemented());

    		fe_name = FEName::FE_RT;

    		base_values_count = 2;
    		n_dofs_1d[0] = fe->degree+1; //Qk+1
    		n_dofs_1d[1] = fe->degree; //Qk

        	for (int c=0; c<vector_n_components; c++)
        		for (int d=0; d<dim; d++)
        		{
        			if (c==d)
        				index_map[mindex(c,d)] = 0;
        			else
        				index_map[mindex(c,d)] = 1;
        		}
    	}
    	else if(dynamic_cast<const FE_Q<dim> *>(fe))
    	{
        	for (int c=0; c<vector_n_components; c++)
        		for (int d=0; d<dim; d++)
        			index_map[mindex(c,d)] = 0;

    		base_values_count = 1;
    		n_dofs_1d[0] = fe->degree+1;

    		fe_name = FEName::FE_Q_TP;
    	}
    	else
    	{
    		Assert (false, ExcNotImplemented());
    	}


        fe_degree = fe->degree; //Note that for RT, this is max degree across all components/dimensions
        n_q_points_1d = quad.size();
        n_q_points      = Utilities::fixed_power<dim>(n_q_points_1d);
        //For tensor product FE like Lagrangian or RT, this division is exact
        dofs_per_component_on_cell = fe_in.dofs_per_cell/vector_n_components;

        //Required for FE_Q
        std::vector<unsigned int> scalar_lexicographic;
        Point<dim> unit_point;

        //Required for RT
        std::vector<std::vector<Polynomials::Polynomial<double>>> pols; //1-D polynomials
        std::vector<PolynomialSpace<1>> polyspace;
        std::vector<FullMatrix<double>> tensor_inv_nodal; //tensor form of inverse nodal matrix

        if (FEName::FE_Q_TP == fe_name)
        {
        	//Similar to internal_reinit_scalar
            Assert(fe->n_components() == 1,ExcMessage("Expected a scalar element"));

            scalar_lexicographic = lexicographic_renumber(fe_in, base_element_number);

            if (fe->has_support_points())
              unit_point = fe->get_unit_support_points()[scalar_lexicographic[0]];
            Assert(fe->dofs_per_cell == 0 ||
                   std::abs(fe->shape_value(scalar_lexicographic[0],
                                            unit_point)-1) < 1e-13,
                   ExcInternalError("Could not decode 1D shape functions for the "
                                    "element " + fe->get_name()));
        }
        else if (FEName::FE_RT == fe_name)
        {
        	const FE_RaviartThomas<dim> * fe_rt = dynamic_cast<const FE_RaviartThomas<dim> *>(fe);
        	raviart_thomas_lexicographic_renumber(fe_rt);

        	  pols = PolynomialsRaviartThomas<dim>::create_polynomials (fe_degree-1);
        	  polyspace.reserve(base_values_count);
        	  for (int j=0; j<base_values_count; j++)
        	  {
        		  polyspace.emplace_back(PolynomialSpace<1>(pols[j]));
        		  Assert(polyspace[j].n() == n_dofs_1d[j],ExcMessage("Polynomial space size mismatch with expected dofs"));
        	 }

        	  //For n = 2, the last component helps to obtain the required tensor product form
        	  //for RT N matrices
        	  //Reorder and extract last component from tra(inverse nodal matrix)
          	  const FullMatrix<double> &inv_nodeM = fe_rt->get_inverse_node_matrix();

          	  std::vector<unsigned int> row_index_set(dofs_per_component_on_cell);
          	  std::vector<unsigned int> col_index_set(dofs_per_component_on_cell);

          	  int i = (dim-1)*dofs_per_component_on_cell;
          	  for (auto &index : row_index_set)
          	        index = i++;

          	  i = (dim-1)*dofs_per_component_on_cell;
          	  for (auto &index : col_index_set)
          	        index = lexicographic_numbering[i++];

          	  FullMatrix<double> T(dofs_per_component_on_cell,dofs_per_component_on_cell);
          	  T.extract_submatrix_from(inv_nodeM,row_index_set,col_index_set);

          	  //We know that a tensor product structure exists in T as:
          	  // tra(T) = B \otimes A
          	  // where A = kxk matrix
          	  // B = (k+1)x(k+1) matrix
          	  //e.g. for k=1, (3x3)\otimes (2x2) structrue
          	  FullMatrix<double> tempM(fe_degree,fe_degree);
          	  FullMatrix<double> A(fe_degree,fe_degree), B(fe_degree+1,fe_degree+1);

          	  std::vector<int> v_row(fe_degree), v_col(fe_degree);
          	  std::iota(std::begin(v_row),std::end(v_row),0);
          	  A.extract_submatrix_from(T,v_row,v_row);

          	  B(0,0) = 1.0;

          	  for (int i=0; i<(fe_degree+1);i++)
          	  {
          		  for (int j=0; j<(fe_degree+1); j++)
          		  {
          			  if (i==0 && j==0)
          				  continue;

          			  std::iota(std::begin(v_row),std::end(v_row),i*fe_degree);
          			  std::iota(std::begin(v_col),std::end(v_col),j*fe_degree);
          			  tempM.extract_submatrix_from(T,v_row,v_col);
          			  double val = std::numeric_limits<double>::lowest();
          			  for (int x=0; x<fe_degree; x++)
          				  for (int y=0; y<fe_degree; y++)
          					  if (std::fabs(A(x,y)) > 10e-6) //Avoid division by ~zero, chosen by experimentation
          						  val = std::max(val,tempM(x,y)/A(x,y)); //too small values of tempM should not bias result

          			  B(j,i) = val; //transposed
          		  }
          	  }
          	tempM = A;
          	A.copy_transposed(tempM);

          	//push order is important for pop later
          	tensor_inv_nodal.push_back(B);
          	tensor_inv_nodal.push_back(A);

#if _UT__
          	//Only for debugging
    		std::cout<<"A Matrix is"<<std::endl;
    		for (unsigned int i=0; i<fe_degree; i++)
    		{
    			for (unsigned int j=0; j<fe_degree; j++)
    			{
    				std::cout <<"   "<<tensor_inv_nodal[1](i,j);
    			}
    			std::cout<<std::endl;
    		}
    		std::cout<<std::endl<<std::endl;

    		std::cout<<"B Matrix is"<<std::endl;
    		for (unsigned int i=0; i<fe_degree+1; i++)
    		{
    			for (unsigned int j=0; j<fe_degree+1; j++)
    			{
    				std::cout <<"   "<<tensor_inv_nodal[0](i,j);
    			}
    			std::cout<<std::endl;
    		}
    		std::cout<<std::endl<<std::endl;
#endif
        }

        this->base_shape_gradients.resize(base_values_count);
        this->base_shape_values.resize(base_values_count);
        this->base_shape_hessians.resize(base_values_count);

        for (int j=0; j<base_values_count; j++)
        {
        	const unsigned int array_size = n_dofs_1d[j]*n_q_points_1d;

        	this->base_shape_gradients[j].resize_fast (array_size);
        	this->base_shape_values[j].resize_fast (array_size);
        	this->base_shape_hessians[j].resize_fast (array_size);

        	if (FEName::FE_RT == fe_name)
        	{
                std::vector<double> p_values;
                std::vector<Tensor<1,1> > p_grads;
                std::vector<Tensor<2,1> > p_grad_grads;
                std::vector<Tensor<3,1> > p_third_derivatives(0);
                std::vector<Tensor<4,1> > p_fourth_derivatives(0);

        		for (unsigned int q=0; q<n_q_points_1d; ++q)
        		{
        			Point<1> q_point = quad.get_points()[q];
        			const unsigned int psize = polyspace[j].n();
        			p_values.resize(psize);
        			p_grads.resize(psize);
        			p_grad_grads.resize(psize);

        			polyspace[j].compute(q_point,p_values,p_grads,p_grad_grads,p_third_derivatives,p_fourth_derivatives);

                	for (unsigned int i=0; i<n_dofs_1d[j]; ++i)
                	{
                		// Dot product of vectors
                		this->base_shape_values[j][i*n_q_points_1d+q] = tensor_inv_nodal[j](i,0)*p_values[0];
                		for (int k=1; k<psize; k++)
                			this->base_shape_values[j][i*n_q_points_1d+q] = this->base_shape_values[j][i*n_q_points_1d+q] + tensor_inv_nodal[j](i,k)*p_values[k];

                		this->base_shape_gradients[j][i*n_q_points_1d+q] = tensor_inv_nodal[j](i,0)*p_grads[0][0];
                		for (int k=1; k<psize; k++)
                		     this->base_shape_gradients[j][i*n_q_points_1d+q] = this->base_shape_gradients[j][i*n_q_points_1d+q] + tensor_inv_nodal[j](i,k)*p_grads[k][0];

                		//TODO: Yet to be verified for hessians
                		this->base_shape_hessians[j][i*n_q_points_1d+q] = tensor_inv_nodal[j](i,0)*p_grad_grads[0][0][0];
                		for (int k=1; k<psize; k++)
                		     this->base_shape_hessians[j][i*n_q_points_1d+q] = this->base_shape_hessians[j][i*n_q_points_1d+q] + tensor_inv_nodal[j](i,k)*p_grad_grads[k][0][0];
                	}
        		}
        	}

        	if (FEName::FE_Q_TP == fe_name)
        	{
            	FE_Q<1> temp_fe(n_dofs_1d[j]-1); //fe_degree = dofs_1d-1
            	std::vector<unsigned int >scalar_lexicographic_temp = temp_fe.get_poly_space_numbering_inverse();

            	for (unsigned int i=0; i<n_dofs_1d[j]; ++i)
            	{
            		// need to reorder from hierarchical to lexicographic to get the
            		// DoFs correct
            		const unsigned int my_i = scalar_lexicographic_temp[i];
            		for (unsigned int q=0; q<n_q_points_1d; ++q)
            		{
            			Point<1> q_point = quad.get_points()[q];

            			this->base_shape_values[j][i*n_q_points_1d+q] = temp_fe.shape_value(my_i,q_point);
            			this->base_shape_gradients[j][i*n_q_points_1d+q] = temp_fe.shape_grad(my_i,q_point)[0];
            			this->base_shape_hessians[j][i*n_q_points_1d+q] = temp_fe.shape_grad_grad(my_i,q_point)[0][0];
            		}
            	}
        	}
        }

        shape_values_vec.resize(vector_n_components);
    	shape_gradients_vec.resize(vector_n_components);
        shape_hessians_vec.resize(vector_n_components);

    	for (int c=0; c<vector_n_components; c++)
        {
        	for (int d=0; d<dim; d++)
        	{
    		  	shape_gradients_vec[c][d] = this->base_shape_gradients[index_map[mindex(c,d)]].begin();
    		   	shape_values_vec[c][d] = this->base_shape_values[index_map[mindex(c,d)]].begin();
    	    	shape_hessians_vec[c][d] = this->base_shape_hessians[index_map[mindex(c,d)]].begin();
        	}
        }
    }
    // end of functions for ShapeInfo
  } // end of namespace MatrixFreeFunctions
} // end of namespace internal


DEAL_II_NAMESPACE_CLOSE

#endif

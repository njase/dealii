/*
 * fe_evaluation_gen.h
 *
 *  Created on: Dec 11, 2017
 *      Author: smehta
 */

#ifndef FE_EVALUATION_GEN_H_
#define FE_EVALUATION_GEN_H_


#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_raviart_thomas.h>

DEAL_II_NAMESPACE_OPEN


	//TODO:
	//I would like to have traits for something like FESystem<dim>, but that seems bit difficult
	//Right now focus on RT



	class FE_TaylorHood{
	}; //empty class, only for getting the typename


	///////some traits
	template <typename T, int dim>
	struct get_n_comp
	{
		static constexpr int n_components = 1;
	};


	//TODO: Currently these specializations provide info only for velocity components
	//Extending their use to also pressure component is TBD
	template <int dim>
	struct get_n_comp<FE_RaviartThomas<dim>,dim>
	{
		static constexpr int n_components = dim;
	};

	template <int dim>
	struct get_n_comp<FE_TaylorHood,dim>
	{
		static constexpr int n_components = dim;
	};


	//Get FE Data info in a static manner. The FE object can provide this (and much more) info
	//at runtime
	template <typename FEType, int dim, int dir, int base_fe_degree, int c>
	struct get_FEData
	{
		static constexpr int max_fe_degree = base_fe_degree;
		static constexpr bool isIsotropic = true; //FE_Q in all directions is e.g. isotropic
		static constexpr int fe_degree_for_component = base_fe_degree;
		static constexpr unsigned int dofs_per_cell = base_fe_degree+1;
	};


	//TODO: Currently these specializations provide info only for velocity components
	//Extending their use to also pressure component is TBD
	template <int dim, int dir, int base_fe_degree, int c>
	struct get_FEData<FE_RaviartThomas<dim>, dim, dir, base_fe_degree, c>
	{
		static constexpr int max_fe_degree = base_fe_degree+1;
		static constexpr bool isIsotropic = false;
		static constexpr int fe_degree_for_component = ((dir == c) ? base_fe_degree+1 : base_fe_degree);
		static constexpr unsigned int dofs_per_cell =
					Utilities::fixed_int_power<base_fe_degree+1,dim-1>::value*dim*(base_fe_degree+2);
	};

	//Qk,Q(k-1) element = FE_TaylorHood
	template <int dim, int dir, int base_fe_degree, int c>
	struct get_FEData<FE_TaylorHood, dim, dir, base_fe_degree, c>
	{
		static constexpr int max_fe_degree = base_fe_degree;
		static constexpr bool isIsotropic = true;
		static constexpr int fe_degree_for_component = base_fe_degree;
		static constexpr unsigned int dofs_per_cell =
					Utilities::fixed_int_power<base_fe_degree+1,dim>::value*dim/*=n_components*/;
	};

DEAL_II_NAMESPACE_CLOSE





#endif /* FE_EVALUATION_GEN_H_ */

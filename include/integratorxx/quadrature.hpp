#pragma once

#include <tuple>

namespace IntegratorXX {

// Forward decl of quadrature traits
template <typename Quadrature>
struct quadrature_traits;


/**
 *  \brief Base class for Quadrature.
 */
template <typename PointContainer, typename WeightContainer>
class QuadratureBase {

public:

  using point_container  = PointContainer;
  using weight_container = WeightContainer;

  using point_type  = typename point_container::value_type;
  using weight_type = typename weight_container::value_type;

protected:

  point_container  points_;
  weight_container weights_;

  QuadratureBase( const point_container& p, const weight_container& w ):
    points_(p), weights_(w) { }
  QuadratureBase( point_container&& p, weight_container&& w ):
    points_(std::move(p)), weights_(std::move(w)) { }

public:

  const auto& points()  const { return points_; }
  auto&       points()        { return points_; }
  const auto& weights() const { return weights_; }
  auto&       weights()       { return weights_; }

  size_t npts() const { return points_.size(); }

};

/**
 *  @brief Quadrature base class: Provides minimal storage and user interface 
 *  for quadrature manipulation
 *
 *  @tparam DerivedQuadrature Implemented quadrature type, must admit template
 *  specialization for quadrature_traits
 */
template <typename DerivedQuadrature>
class Quadrature : public
  QuadratureBase< 
    typename quadrature_traits<DerivedQuadrature>::point_container,
    typename quadrature_traits<DerivedQuadrature>::weight_container
  > {

private:

  using derived_traits = quadrature_traits<DerivedQuadrature>;

public:

  using point_type  = typename derived_traits::point_type;
  using weight_type = typename derived_traits::weight_type;

  using point_container  = typename derived_traits::point_container;
  using weight_container = typename derived_traits::weight_container;

private:

  using base_type = QuadratureBase< point_container, weight_container >;

  template <typename... Args>
  inline static constexpr auto 
    generate( Args&&... args ) {
    return derived_traits::generate( std::forward<Args>(args)... );
  }


protected:

  using quadrature_return_type =
    std::tuple<point_container,weight_container>;

  Quadrature( const quadrature_return_type& q ) :
    base_type( std::get<0>(q), std::get<1>(q) ) { }

  Quadrature( quadrature_return_type&& q ) :
    base_type( std::move(std::get<0>(q)), std::move(std::get<1>(q)) ) { }

public:

  template <typename... Args>
  Quadrature( Args&&... args ) :
    Quadrature( generate( std::forward<Args>(args)... ) ) { } 

  Quadrature( const Quadrature& ) = default;
  Quadrature( Quadrature&& ) noexcept = default;

};


}

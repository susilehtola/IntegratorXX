#pragma once

namespace IntegratorXX {
namespace WomersleyGrids {

/**
 *  \brief Womersley Quadrature specification for index 7 grid with 32 points
 * 
 */
template <typename T>
struct womersley_32 {

  static constexpr std::array<cartesian_pt_t<T>,32> points = {
     0.0000000000000000e+00,      0.0000000000000000e+00,      1.0000000000000000e+00,
     6.9538652093116349e-01,      0.0000000000000000e+00,     -7.1863592069089655e-01,
     4.4812457777580794e-01,     -8.2600133339618187e-01,      3.4191542817045201e-01,
    -8.7295687563333738e-02,      5.7244915338765989e-01,     -8.1527997014436471e-01,
     8.3307129775302080e-01,      2.7697026927927648e-01,      4.7883158082509841e-01,
    -8.9148423292088941e-01,     -2.3223519351146743e-01,     -3.8900215596850451e-01,
     9.6409323132612953e-02,     -5.7496506631784772e-01,      8.1247794734824930e-01,
     5.2737619996567575e-01,      5.5670430023408302e-01,     -6.4183694643627642e-01,
    -4.4670986063083357e-01,      8.2348022586910175e-01,     -3.4975794203670996e-01,
    -7.7726124523122497e-01,     -6.2568756084161559e-01,      6.6181816760159634e-02,
    -6.1249028125957472e-01,      5.1046537441687473e-03,      7.9046163592720875e-01,
    -7.9957492931843521e-01,      5.2792968453428268e-01,      2.8630434958790285e-01,
     6.0892136694312693e-01,     -1.8817404300813548e-01,      7.7058763188756552e-01,
    -9.6570724354131832e-01,     -4.0605365894881208e-02,      2.5643853850849235e-01,
     9.8767382061914288e-01,      6.0819801795774113e-02,     -1.4422682057477318e-01,
     7.4666416333048102e-01,     -5.3960165849434671e-01,     -3.8900215596850246e-01,
    -5.2106602999677853e-01,     -5.5631572732793300e-01,      6.4730441363472035e-01,
     3.4241517661034016e-01,      8.7468184934152282e-01,      3.4305030135448522e-01,
    -3.1762246423879614e-01,      5.6602154586619291e-01,      7.6074672514978336e-01,
    -3.5014006372421347e-01,     -8.7254931472059183e-01,     -3.4067525468667270e-01,
     2.8279525504184533e-01,     -9.2084065543571381e-01,     -2.6847594123597485e-01,
     7.7559219605253016e-01,      6.2866190271795430e-01,     -5.6929407984406405e-02,
     1.1532072773686707e-01,     -1.1760443809991116e-02,     -9.9325868821552799e-01,
     8.7000996112490758e-01,     -4.3214381572928218e-01,      2.3734866772403251e-01,
    -2.9023984143243164e-01,      9.2038222648956303e-01,      2.6202555525633703e-01,
    -8.5142492584132901e-01,      3.9902421121264259e-01,     -3.4037519669067506e-01,
    -4.2728863008225687e-01,     -4.7776590042344924e-01,     -7.6757030361719891e-01,
    -5.2523128678993669e-01,      1.4499550434384748e-01,     -8.3851559263800257e-01,
    -1.7745811129773031e-01,     -9.4578541754828238e-01,      2.7202676832926526e-01,
     2.0796878042673947e-01,     -5.6436010110055301e-01,     -7.9890341259353004e-01,
     1.7745811129772868e-01,      9.4578541754828227e-01,     -2.7202676832926620e-01,
     3.2580735512709053e-01,      5.0578085119591965e-01,      7.9877111734752959e-01
};


static constexpr auto weights = detail::create_array<32, T>(4.0 * M_PI / 32.0);
};
}  // namespace WomersleyGrids
}  // namespace IntegratorXX
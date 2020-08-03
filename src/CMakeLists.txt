
add_subdirectory(libpsc)

macro(add_psc_executable name)
  add_executable(${name} ${name}.cxx)
  target_link_libraries(${name} psc)
endmacro(add_psc_executable)

add_psc_executable(psc_Isotropic_AW_test1)
add_psc_executable(psc_flatfoil_yz)
add_psc_executable(psc_electron_project)
#add_psc_executable(psc_whistler)
#add_psc_executable(psc_my_case_particles)
#add_psc_executable(psc_harris_xz)
#add_psc_executable(psc_test_twoparticles)

if(USE_VPIC)
  add_psc_executable(psc_harris_xz)
endif()

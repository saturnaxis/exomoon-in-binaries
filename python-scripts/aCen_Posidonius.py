import posidonius
import rebound
import numpy as np
import argparse

def calculate_spin(angular_frequency, inclination, obliquity, position, velocity):
    """
    Old version of calculate spin used in all cases in Bolmont et al 2015 except
    for cases 6, 6', 6'', 6''' and 7
    """
    if inclination == 0.:
        # No inclination, spin can already be calculated:
        x = angular_frequency * np.sin(obliquity) # zero if there is no obliquity
        y = 0.
        z = angular_frequency * np.cos(obliquity)
    else:
        # Calculation of orbital angular momentum (without mass and in AU^2/day)
        horb_x = position.y() * velocity.z() - position.z() * velocity.y()
        horb_y = position.z() * velocity.x() - position.x() * velocity.z()
        horb_z = position.x() * velocity.y() - position.y() * velocity.x()
        horbn = np.sqrt(np.power(horb_x, 2) + np.power(horb_y, 2) + np.power(horb_z, 2))
        # Spin taking into consideration the inclination:
        x = angular_frequency * (horb_x / (horbn * np.sin(inclination))) * np.sin(obliquity+inclination)
        y = angular_frequency * (horb_y / (horbn * np.sin(inclination))) * np.sin(obliquity+inclination)
        z = angular_frequency * np.cos(obliquity+inclination)
    return posidonius.Axes(x, y, z)

def Orb2Cart(x,m,mp,com):
    temp_sim = rebound.Simulation()
    temp_sim.integrator = "ias15"
    temp_sim.units = ('days', 'AU', 'Msun')
    temp_sim.add(m=m)
    temp_sim.add(m=mp,a=x[0],e=x[1],inc=np.radians(x[2]),omega=np.radians(x[3]),Omega=np.radians(x[4]),M=np.radians(x[5]))
    temp_ps = temp_sim.particles
    if com == "F":
        temp = [mp,temp_ps[1].x,temp_ps[1].y,temp_ps[1].z,temp_ps[1].vx,temp_ps[1].vy,temp_ps[1].vz]
        return temp
    else:
        temp_sim.move_to_com()
        temp_0 = [m,temp_ps[0].x,temp_ps[0].y,temp_ps[0].z,temp_ps[0].vx,temp_ps[0].vy,temp_ps[0].vz]
        temp_1 = [mp,temp_ps[1].x,temp_ps[1].y,temp_ps[1].z,temp_ps[1].vx,temp_ps[1].vy,temp_ps[1].vz]
        return temp_0, temp_1

def get_Cartesian_IC(star,L_star,a_sat,MA_sat):

    #MASSES
    if star == 'A':
        M_A = 1.133
        M_B = 0.972
    else:
        M_A = 0.972
        M_B = 1.133
    M_E = 3.0035e-6
    m_p = M_E
    m_sat = 3.69396868e-8 #luna's mass in solar masses

    

    #RADII
    
    #R_p = 0.000477894503 #Radius of Jupiter in AU
    R_E = 4.26352e-5 #Radius of Earth in AU
    R_p = R_E 
    
    #SEMIS

    a_p = np.sqrt(L_star)
    R_H = a_p*((m_p+m_sat)/(3.*M_A))**(1./3.)#hill radius of planet
    e_p = 1.25*(a_p/23.78)*(0.524/(1.-0.524**2))

    temp_state = [a_sat,0,0.,0,0,MA_sat] #satellite 
    Cart_pl, Cart_sat = Orb2Cart(temp_state,m_p,m_sat,"T") #using the planet-moon barycenter as the origin
    
    temp_state = [a_p,e_p,0.,0.,0.,0.]#planet
    Cart_pCOM = Orb2Cart(temp_state,M_A,m_p+m_sat,"F")#returns planet as if it was at the center (indepdendence of origin for coordinate system)
    Cart_B = Orb2Cart([23.7,0.524,0,0,0,0],M_A,M_B,"F")#returns star B
    
    for i in xrange(1,len(Cart_sat)):
        Cart_pl[i] += Cart_pCOM[i]
        Cart_sat[i] += Cart_pCOM[i]
    
    return Cart_pl, Cart_sat, Cart_pCOM, Cart_B


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_filename', action='store', help='Filename where the initial snapshot will be stored (e.g., universe_integrator.json)')
    parser.add_argument('host_star',action='store',help='Host star the planet is orbiting (A or B)')
    args = parser.parse_args()
    filename = args.output_filename
    #filename = posidonius.constants.BASE_DIR+"target/case3.json"
    host = args.host_star
    #Get Cartesian Initial conditions
    if host == 'B':
        L_star = 0.5
    else:
        L_star = 1.519
    
    
    
    M_E = 3.0035e-6
    R_E = 4.26352e-5
    G = (2.*np.pi/365.25)**2
    a_M = 8.64*R_E                             # semi-major axis (in AU)  10 R_E
    T_M = np.sqrt(a_M**3/(G*1.0123*M_E))
    n_M = 2.*np.pi/T_M

    initial_time = 0*365.25 # time [days] where simulation starts
    time_step = T_M/20. # days
    Cart_pl, Cart_sat, Cart_pCOM, Cart_B = get_Cartesian_IC(host,L_star,a_M,0)
    
    time_limit   = 365.25 * 1.0e8 # days
    historic_snapshot_period = 100.*365.25 # days
    recovery_snapshot_period = 100.*historic_snapshot_period # days
    consider_effects = posidonius.ConsiderEffects({
        "tides": True,
        "rotational_flattening": False,
        "general_relativity": False,
        "disk": False,
        "wind": False,
        "evolution": False,
    })
    universe = posidonius.Universe(initial_time, time_limit, time_step, recovery_snapshot_period, historic_snapshot_period, consider_effects)
    if host == 'B':
        star_mass = 0.972 # Solar masses
        star_radius_factor = 0.86
    else:
        star_mass = 1.133 # Solar masses
        star_radius_factor = 1.223
    # [start correction] -------------------------------------------------------
    # To reproduce Bolmont et al. 2015:
    #   Mercury-T was using R_SUN of 4.67920694e-3 AU which is not the IAU accepted value
    #   thus, to reproduce Mercury-T results the star radius factor should be slighly modified:
    star_radius_factor = star_radius_factor*(4.67920694e-3 / posidonius.constants.R_SUN)
    # [end correction] ---------------------------------------------------------
    star_radius = star_radius_factor * posidonius.constants.R_SUN
    star_radius_of_gyration = np.sqrt(0.07) #solar moi
    
    star_position = posidonius.Axes(0., 0., 0.)
    star_velocity = posidonius.Axes(0., 0., 0.)

    
    # Initialization of stellar spin
    star_rotation_period = 70.0 # hours
    star_angular_frequency = posidonius.constants.TWO_PI/(star_rotation_period/24.) # days^-1
    star_spin = posidonius.Axes(0., 0., star_angular_frequency)
    star_tides_parameters = {
        "dissipation_factor_scale": 1.0,
        "dissipation_factor": 4.992*3.845764e-2,
        "love_number": 0.03,
    }
    star_tides = posidonius.effects.tides.OrbitingBody(star_tides_parameters)
    #star_tides = posidonius.effects.tides.Disabled()
    #star_rotational_flattening_parameters = {"love_number": star_tides_parameters["love_number"]}
    #star_rotational_flattening = posidonius.effects.rotational_flattening.OrbitingBody(star_rotational_flattening_parameters)
    star_rotational_flattening = posidonius.effects.rotational_flattening.Disabled()
    
    star_general_relativity = posidonius.effects.general_relativity.Disabled()
    star_wind = posidonius.effects.wind.Disabled()
    star_disk = posidonius.effects.disk.Disabled()
    star_evolution = posidonius.NonEvolving()
    #
    star = posidonius.Particle(star_mass, star_radius, star_radius_of_gyration, star_position, star_velocity, star_spin)
    star.set_tides(star_tides)
    star.set_rotational_flattening(star_rotational_flattening)
    star.set_general_relativity(star_general_relativity)
    star.set_wind(star_wind)
    star.set_disk(star_disk)
    star.set_evolution(star_evolution)
    universe.add_particle(star)
    
    ############################################################################
    planet_mass_factor = 1.0
    # [start correction] -------------------------------------------------------
    # To reproduce Bolmont et al. 2015:
    #   Mercury-T was using planet_mass as 3.00e-6 M_SUN and that's not exactly 1 M_EARTH (as accepted by IAU)
    #   thus, to reproduce Mercury-T results the mass factor should be slighly modified:
    planet_mass_factor = planet_mass_factor * (3.00e-6 / posidonius.constants.M_EARTH) # 0.999000999000999
    # [end correction] ---------------------------------------------------------
    planet_mass = planet_mass_factor * posidonius.constants.M_EARTH # Solar masses (3.0e-6 solar masses = 1 earth mass)

    # Earth-like => mass-radius relationship from Fortney 2007
    planet_radius_factor = posidonius.tools.mass_radius_relation(planet_mass_factor, planet_mass_type='factor', planet_percent_rock=0.70)
    # [start correction] -------------------------------------------------------
    # To reproduce Bolmont et al. 2015:
    #   Mercury-T defined a different M2EARTH from the IAU accepted value
    #   and M2EARTH was used to compute planet_radius_factor, thus to reproduce
    #   Mercury-T results the planet_radius_factor has to be corrected:
    planet_radius_factor = planet_radius_factor * 0.999756053794 # 1.0097617465214679
    # [end correction] ---------------------------------------------------------
    planet_radius = planet_radius_factor * posidonius.constants.R_EARTH
    planet_radius_of_gyration = 5.75e-01
    
     
    planet_position = posidonius.Axes(Cart_pl[1], Cart_pl[2], Cart_pl[3])
    planet_velocity = posidonius.Axes(Cart_pl[4], Cart_pl[5], Cart_pl[6])

    #////// Initialization of planetary spin
    planet_obliquity = 0. * posidonius.constants.DEG2RAD # 0.2 rad
    planet_rotation_period = 5. # hours
    planet_angular_frequency = posidonius.constants.TWO_PI/(planet_rotation_period/24.) # days^-1
    # Pseudo-synchronization period
    #planet_keplerian_orbital_elements = posidonius.calculate_keplerian_orbital_elements(planet_mass, planet_position, planet_velocity, masses=[star_mass], positions=[star_position], velocities=[star_velocity])
    #planet_semi_major_axis = planet_keplerian_orbital_elements[0]
    #planet_eccentricity = planet_keplerian_orbital_elements[2]
    #planet_semi_major_axis = a
    #planet_eccentricity = e
    #planet_pseudo_synchronization_period = posidonius.calculate_pseudo_synchronization_period(planet_semi_major_axis, planet_eccentricity, star_mass, planet_mass) # days
    #planet_angular_frequency = posidonius.constants.TWO_PI/(planet_pseudo_synchronization_period) # days^-1
    #planet_keplerian_orbital_elements = posidonius.calculate_keplerian_orbital_elements(planet_mass, planet_position, planet_velocity, masses=[star_mass], positions=[star_position], velocities=[star_velocity])
    planet_inclination = 0.
    planet_spin = calculate_spin(planet_angular_frequency, planet_inclination, planet_obliquity, planet_position, planet_velocity)
    Q_p = 12.
    Omega_p = 2.*np.pi/(planet_rotation_period/24.)
    k2pdelta = 0.299*100./86400.#2.465278e-3 # Terrestrial planets (no gas)
    planet_tides_parameters = {
        "dissipation_factor_scale": 1.0,
        "dissipation_factor": 2. * posidonius.constants.K2 * k2pdelta/(3. * np.power(planet_radius, 5)),
        "love_number": 0.299,
    }
    planet_tides = posidonius.effects.tides.CentralBody(planet_tides_parameters)
    #planet_tides = posidonius.effects.tides.OrbitingBody(planet_tides_parameters)
    #planet_tides = posidonius.effects.tides.Disabled()
    #
    #planet_rotational_flattening_parameters = {"love_number": planet_tides_parameters["love_number"]}
    #planet_rotational_flattening = posidonius.effects.rotational_flattening.CentralBody(planet_rotational_flattening_parameters)
    #planet_rotational_flattening = posidonius.effects.rotational_flattening.OrbitingBody(planet_rotational_flattening_parameters)
    planet_rotational_flattening = posidonius.effects.rotational_flattening.Disabled()
    
    planet_general_relativity = posidonius.effects.general_relativity.Disabled()
    planet_wind = posidonius.effects.wind.Disabled()
    planet_disk = posidonius.effects.disk.Disabled()
    planet_evolution = posidonius.NonEvolving()
    #
    planet = posidonius.Particle(planet_mass, planet_radius, planet_radius_of_gyration, planet_position, planet_velocity, planet_spin)
    planet.set_tides(planet_tides)
    planet.set_rotational_flattening(planet_rotational_flattening)
    planet.set_general_relativity(planet_general_relativity)
    planet.set_wind(planet_wind)
    planet.set_disk(planet_disk)
    planet.set_evolution(planet_evolution)
    universe.add_particle(planet)
    
    moon_mass_factor = 0.0123
    # [start correction] -------------------------------------------------------
    # To reproduce Bolmont et al. 2015:
    #   Mercury-T was using planet_mass as 3.00e-6 M_SUN and that's not exactly 1 M_EARTH (as accepted by IAU)
    #   thus, to reproduce Mercury-T results the mass factor should be slighly modified:
    moon_mass_factor = moon_mass_factor * (3.00e-6 / posidonius.constants.M_EARTH) # 0.999000999000999
    # [end correction] ---------------------------------------------------------
    moon_mass = moon_mass_factor * posidonius.constants.M_EARTH # Solar masses (3.0e-6 solar masses = 1 earth mass)

    # Earth-like => mass-radius relationship from Fortney 2007
    #planet_radius_factor = posidonius.tools.mass_radius_relation(planet_mass_factor, planet_mass_type='factor', planet_percent_rock=0.70)
    # [start correction] -------------------------------------------------------
    # To reproduce Bolmont et al. 2015:
    #   Mercury-T defined a different M2EARTH from the IAU accepted value
    #   and M2EARTH was used to compute planet_radius_factor, thus to reproduce
    #   Mercury-T results the planet_radius_factor has to be corrected:
    moon_radius_factor = 0.27 * 0.999756053794 # 1.0097617465214679
    # [end correction] ---------------------------------------------------------
    moon_radius = planet_radius_factor * posidonius.constants.R_EARTH
    moon_radius_of_gyration = np.sqrt(0.39)#5.75e-01
    
    moon_position = posidonius.Axes(Cart_sat[1], Cart_sat[2], Cart_sat[3])
    moon_velocity = posidonius.Axes(Cart_sat[4], Cart_sat[5], Cart_sat[6])
    
    #////// Initialization of planetary spin
    moon_obliquity = 0 * posidonius.constants.DEG2RAD # 0.2 rad
    moon_rotation_period = 10. # hours
    #Omega_m = 2.*np.pi/(planet_rotation_period/24.)
    moon_angular_frequency = posidonius.constants.TWO_PI/(moon_rotation_period/24.) # days^-1
    # Pseudo-synchronization period
    #planet_keplerian_orbital_elements = posidonius.calculate_keplerian_orbital_elements(planet_mass, planet_position, planet_velocity, masses=[star_mass], positions=[star_position], velocities=[star_velocity])
    #planet_semi_major_axis = planet_keplerian_orbital_elements[0]
    #planet_eccentricity = planet_keplerian_orbital_elements[2]
    #planet_semi_major_axis = a
    #planet_eccentricity = e
    #planet_pseudo_synchronization_period = posidonius.calculate_pseudo_synchronization_period(planet_semi_major_axis, planet_eccentricity, star_mass, planet_mass) # days
    #planet_angular_frequency = posidonius.constants.TWO_PI/(planet_pseudo_synchronization_period) # days^-1
    #moon_keplerian_orbital_elements = posidonius.calculate_keplerian_orbital_elements(moon_mass, planet_position, planet_velocity, masses=[star_mass], positions=[star_position], velocities=[star_velocity])
    moon_inclination = 0.
    moon_spin = calculate_spin(moon_angular_frequency, moon_inclination, moon_obliquity, moon_position, moon_velocity)

    Q_m = 37.5
    k2mdelta = 0.3*2.465278e-3 # Terrestrial planets (no gas)
    moon_tides_parameters = {
        "dissipation_factor_scale": 0.0,
        "dissipation_factor": 2. * posidonius.constants.K2 * k2mdelta/(3. * np.power(moon_radius, 5)),
        "love_number": 0.024,
    }
    moon_tides = posidonius.effects.tides.OrbitingBody(moon_tides_parameters)
    #moon_tides = posidonius.effects.tides.Disabled()

    #moon_rotational_flattening_parameters = {"love_number": moon_tides_parameters["love_number"]}
    #moon_rotational_flattening = posidonius.effects.rotational_flattening.CentralBody(moon_rotational_flattening_parameters)
    #moon_rotational_flattening = posidonius.effects.rotational_flattening.OrbitingBody(moon_rotational_flattening_parameters)
    moon_rotational_flattening = posidonius.effects.rotational_flattening.Disabled()
    
    moon_general_relativity = posidonius.effects.general_relativity.Disabled()
    moon_wind = posidonius.effects.wind.Disabled()
    moon_disk = posidonius.effects.disk.Disabled()
    moon_evolution = posidonius.NonEvolving()
    #
    moon = posidonius.Particle(moon_mass, moon_radius, moon_radius_of_gyration, moon_position, moon_velocity, moon_spin)
    moon.set_tides(moon_tides)
    moon.set_rotational_flattening(moon_rotational_flattening)
    moon.set_general_relativity(moon_general_relativity)
    moon.set_wind(moon_wind)
    moon.set_disk(moon_disk)
    moon.set_evolution(moon_evolution)
    universe.add_particle(moon)
    
    if host == 'B':
        star_mass_B = 1.133 # Solar masses
        star_radius_factor = 1.223
    else:
        star_mass_B = 0.972 # Solar masses
        star_radius_factor = 0.86
    # [start correction] -------------------------------------------------------
    # To reproduce Bolmont et al. 2015:
    #   Mercury-T was using R_SUN of 4.67920694e-3 AU which is not the IAU accepted value
    #   thus, to reproduce Mercury-T results the star radius factor should be slighly modified:
    star_radius_factor = star_radius_factor*(4.67920694e-3 / posidonius.constants.R_SUN)
    # [end correction] ---------------------------------------------------------
    star_radius = star_radius_factor * posidonius.constants.R_SUN
    star_radius_of_gyration = 4.41e-01 # Brown dwarf

    star_position = posidonius.Axes(Cart_B[1], Cart_B[2], Cart_B[3])
    star_velocity = posidonius.Axes(Cart_B[4], Cart_B[5], Cart_B[6])
    
    # Initialization of stellar spin
    star_rotation_period = 70.0 # hours
    star_angular_frequency = posidonius.constants.TWO_PI/(star_rotation_period/24.) # days^-1
    star_spin = posidonius.Axes(0., 0., star_angular_frequency)

    star_tides_parameters = {
        "dissipation_factor_scale": 0.0,
        "dissipation_factor": 4.992*3.845764e-2,
        "love_number": 0.03,
    }
    star_tides = posidonius.effects.tides.Disabled()
    star_rotational_flattening = posidonius.effects.rotational_flattening.Disabled()
    star_general_relativity = posidonius.effects.general_relativity.Disabled()
    star_wind = posidonius.effects.wind.Disabled()
    star_disk = posidonius.effects.disk.Disabled()
    star_evolution = posidonius.NonEvolving()
    #
    star = posidonius.Particle(star_mass_B, star_radius, star_radius_of_gyration, star_position, star_velocity, star_spin)
    star.set_tides(star_tides)
    star.set_rotational_flattening(star_rotational_flattening)
    star.set_general_relativity(star_general_relativity)
    star.set_wind(star_wind)
    star.set_disk(star_disk)
    star.set_evolution(star_evolution)
    universe.add_particle(star)
    
    

    whfast_alternative_coordinates="DemocraticHeliocentric"
    #whfast_alternative_coordinates="WHDS"
    #whfast_alternative_coordinates="Jacobi"
    universe.write(filename, integrator="WHFast", whfast_alternative_coordinates=whfast_alternative_coordinates)
    #universe.write(filename, integrator="IAS15")
    #universe.write(filename, integrator="LeapFrog")



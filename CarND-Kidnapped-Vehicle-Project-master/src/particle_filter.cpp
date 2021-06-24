/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  // empirical studies shows that the error rate of x and y
  // reduces from 0.5 to 0.1 as the number of particiles are
  // increased from 2 to 100. After 100, the error rates does
  // not show any improvement.
  // https://knowledge.udacity.com/questions/29851
  if (!is_initialized){
    num_particles = 30; // TODO: Set the number of particles

    // set random engine

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta , std[2]);

    for (int i=0 ; i<num_particles; ++i){
      Particle particle;
      particle.id = i;
      particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
      particle.weight = 1.0;
      particles.push_back(particle);
    }
    cout << "Particles initialized. Count is " << particles.size() << endl;
    weights.resize(num_particles);
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  default_random_engine gen;
  // noise components for the x and y
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for (int i=0 ; i<num_particles; ++i){
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + noise_x(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + noise_y(gen);
      particles[i].theta += noise_theta(gen);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + noise_x(gen);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + noise_y(gen);
      particles[i].theta += yaw_rate * delta_t + noise_theta(gen);
    }
    //cout << "Particle Index:" << i << " x: " << particles[i].x << " y: " << particles[i].y << " theta: " << particles[i].theta << endl;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); ++i) {
    double distance_min = std::numeric_limits<double>::max();

    for (unsigned j = 0; j < predicted.size(); j++ ) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < distance_min) {
        distance_min = distance;
        observations[i].id = predicted[j].id;
      } // end if distance check
    } // end for - predictions
  } // end for - observations
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i=0 ; i < num_particles; ++i){
    // reset weights of every particles
    particles[i].weight = 1.0;
    // Step 1: loop over observations
    vector<LandmarkObs> observations_t;
    for(unsigned int j=0; j < observations.size(); ++j){
      // transform the observations to map coordinates using homogenous transformation matrix
      LandmarkObs observation_t;
      observation_t.id = observations[j].id;
      observation_t.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      observation_t.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
      observations_t.push_back(observation_t);
    } // end loop for observation iteration

    //Step 2: Identify those landmarks and save them in the predicted_landmarks vector.
    vector<LandmarkObs> predicted_landmarks;
    for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[k];
      if ((fabs((particles[i].x - current_landmark.x_f)) <= sensor_range) &&
          (fabs((particles[i].y - current_landmark.y_f)) <= sensor_range)) {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    } //end loop - predicted_landmarks

    // Step 3: associate the closest landmark to each transformed observation.
    dataAssociation(predicted_landmarks, observations_t);

    // Step 4: calculate each particle's final weight
    // as the product of each measurement's Multivariate-Gaussian probability density.
    for(int m=0; m < int(observations_t.size());m++){
      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(observations_t[m].id-1);
      double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double exponent = (pow(observations_t[m].x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2)))
                     + (pow(observations_t[m].y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2)));
      particles[i].weight *= gauss_norm * exp(-exponent);
    }
    // update the weights. TODO: Check if we need to normalize the weights
    weights[i] = particles[i].weight;
    // cleanup the vector<LandmarkObs> so that they are ready for the next particle
    predicted_landmarks.clear();
    observations_t.clear();
  }//end loop - number of particiles
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  // std::discrete_distribution produces random integers on the interval [0, n)
  // where the probability of each individual integer i is
  // weight of the ith integer divided by the sum of all n weights.

  // same concept as that of the sampling wheel
  std::discrete_distribution<> weight_dist(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  for(int i=0; i<num_particles; i++){
    int index = weight_dist(gen);
    resampled_particles.push_back(particles[index]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


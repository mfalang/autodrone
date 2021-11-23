/*
 * Copyright (C) 2012 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/
#include <gazebo/gazebo.hh>
#include <ignition/math.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265


namespace gazebo
{
  class AnimatedBox : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;


        // Setup
        double period = 6.0;
        double step = 0.2;
        double amplitude = 0.05;

        // create the animation
        gazebo::common::PoseAnimationPtr anim(
              // name the animation "test",
              // make it last <period> seconds,
              // and set it on a repeat loop
              new gazebo::common::PoseAnimation("test", period, true));

        gazebo::common::PoseKeyFrame *key;

        double height, roll;

        // Generate animation
        for (double t=0.0; t <= period; t += step){

          double x = (t/period)*2*PI;

          height = amplitude*sin(x);
          roll = atan(amplitude*cos(x));

          key = anim->CreateKeyFrame(t);
          key->Translation(ignition::math::Vector3d(0, 0, height));
          key->Rotation(ignition::math::Quaterniond(roll, 0, 0));
        }

        // // set starting location of the box
        // key = anim->CreateKeyFrame(0);
        // key->Translation(ignition::math::Vector3d(0, 0, 0));
        // key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        // // set waypoint location after 2 seconds
        // key = anim->CreateKeyFrame(2.0);
        // key->Translation(ignition::math::Vector3d(0, 0, 0.025));
        // key->Rotation(ignition::math::Quaterniond(0.1, 0, 0));

        // key = anim->CreateKeyFrame(4.0);
        // key->Translation(ignition::math::Vector3d(0, 0, 0.05));
        // key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        // key = anim->CreateKeyFrame(6.0);
        // key->Translation(ignition::math::Vector3d(0, 0, 0.025));
        // key->Rotation(ignition::math::Quaterniond(-0.1, 0, 0));

        // key = anim->CreateKeyFrame(8.0);
        // key->Translation(ignition::math::Vector3d(0, 0, 0));
        // key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        // key = anim->CreateKeyFrame(10.0);
        // key->Translation(ignition::math::Vector3d(0, 0, -0.025));
        // key->Rotation(ignition::math::Quaterniond(0.1, 0, 0));

        // key = anim->CreateKeyFrame(12.0);
        // key->Translation(ignition::math::Vector3d(0, 0, -0.05));
        // key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        // key = anim->CreateKeyFrame(14.0);
        // key->Translation(ignition::math::Vector3d(0, 0, -0.025));
        // key->Rotation(ignition::math::Quaterniond(-0.1, 0, 0));

        // // set final location equal to starting location
        // key = anim->CreateKeyFrame(16.0);
        // key->Translation(ignition::math::Vector3d(0, 0, 0));
        // key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        // set the animation
        _parent->SetAnimation(anim);
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(AnimatedBox)
}

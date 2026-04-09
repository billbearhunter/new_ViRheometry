//
//  Camera.cpp
//
//  Created by Yonghao Yue on 2019/09/28.
//  Updated 2021/10/13
//  Copyright © 2019 Yonghao Yue. All rights reserved.
//

#include "Camera.h"

Camera::Camera()
: m_EyePoint( Eigen::Vector3d::Zero() )
, m_xVector( Eigen::Vector3d{ 1.0, 0.0, 0.0 } )
, m_yVector( Eigen::Vector3d{ 0.0, 1.0, 0.0 } )
, m_zVector( Eigen::Vector3d{ 0.0, 0.0, 1.0 } )
, m_DistanceToObject( 1.0 )
, m_FocalLength( 0.035 )
, m_ScreenWidth( 0.036 )
, m_ScreenHeight( 0.024 )
{
  
}

void Camera::setEyePoint( const Eigen::Vector3d& in_eyePoint )
{
  m_EyePoint = in_eyePoint;
}

void Camera::setDistanceToObject( const double& in_DistanceToObject )
{
  m_DistanceToObject = in_DistanceToObject;
}

void Camera::setFocalLength( const double& in_FocalLength )
{
  m_FocalLength = in_FocalLength;
}

void Camera::lookAt( const Eigen::Vector3d& in_LookAt, const Eigen::Vector3d& in_Up )
{
  const Eigen::Vector3d arm = in_LookAt - m_EyePoint;
  m_DistanceToObject = arm.norm();
  m_zVector = - arm / m_DistanceToObject;
  
  const double dot_up_z = in_Up.dot( m_zVector );
  m_yVector = in_Up - dot_up_z * m_zVector;
  m_yVector.normalize();
  m_xVector = m_yVector.cross( m_zVector );
}

void Camera::moveInGlobalFrame( const Eigen::Vector3d& in_delta )
{
  m_EyePoint += in_delta;
}

void Camera::moveInLocalFrame( const Eigen::Vector3d& in_delta )
{
  m_EyePoint += in_delta.x() * m_xVector + in_delta.y() * m_yVector + in_delta.z() * m_zVector;
}

void Camera::moveInGlobalFrameFixLookAt( const Eigen::Vector3d& in_delta )
{
  const Eigen::Vector3d lookAtPoint = getLookAtPoint();
  const Eigen::Vector3d up = m_yVector;
  
  moveInGlobalFrame( in_delta );
  lookAt( lookAtPoint, up );
}

void Camera::moveInLocalFrameFixLookAt( const Eigen::Vector3d& in_delta )
{
  const Eigen::Vector3d lookAtPoint = getLookAtPoint();
  const Eigen::Vector3d up = m_yVector;
  
  moveInLocalFrame( in_delta );
  lookAt( lookAtPoint, up );
}

void Camera::rotateCameraInLocalFrameFixLookAt( const double& in_HorizontalAngle )
{
  const Eigen::Vector3d lookAtPoint = getLookAtPoint();
  Eigen::Vector3d arm = m_DistanceToObject * m_zVector;
  
  const Eigen::Vector3d worldUp { 0.0, 1.0, 0.0 };
  
  // rotate around y-axis
  m_xVector = rotateVector( m_xVector, worldUp, in_HorizontalAngle );
  m_yVector = rotateVector( m_yVector, worldUp, in_HorizontalAngle );
  m_zVector = rotateVector( m_zVector, worldUp, in_HorizontalAngle );
  arm = rotateVector( arm, worldUp, in_HorizontalAngle );
  
  m_EyePoint = lookAtPoint + arm;
}

Eigen::Vector3d Camera::getLookAtPoint() const
{
  return m_EyePoint - m_DistanceToObject * m_zVector;
}

Eigen::Vector3d Camera::getEyePoint() const
{
  return m_EyePoint;
}

Eigen::Vector3d Camera::getXVector() const
{
  return m_xVector;
}

Eigen::Vector3d Camera::getYVector() const
{
  return m_yVector;
}

Eigen::Vector3d Camera::getZVector() const
{
  return m_zVector;
}

double Camera::getDistanceToObject() const
{
  return m_DistanceToObject;
}

double Camera::getFocalLength() const
{
  return m_FocalLength;
}

double Camera::getScreenWidth() const
{
  return m_ScreenWidth;
}

double Camera::getScreenHeight() const
{
  return m_ScreenHeight;
}


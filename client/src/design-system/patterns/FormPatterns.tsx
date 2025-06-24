/**
 * Form Patterns
 * Reusable form patterns using the design system
 */

import * as React from 'react'
import { motion } from 'framer-motion'
import { Button, Input, InputGroup, Card, CardHeader, CardTitle, CardContent, CardFooter } from '../components'
import { animations } from '../animations'
import { cn } from '../../utils/utils'

// Login Form Pattern
export const LoginFormPattern = () => {
  const [email, setEmail] = React.useState('')
  const [password, setPassword] = React.useState('')
  const [loading, setLoading] = React.useState(false)
  const [errors, setErrors] = React.useState<{ email?: string; password?: string }>({})
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrors({})
    setLoading(true)
    
    // Simulate validation
    const newErrors: typeof errors = {}
    if (!email) newErrors.email = 'Email is required'
    if (!password) newErrors.password = 'Password is required'
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      setLoading(false)
      return
    }
    
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setLoading(false)
  }
  
  return (
    <Card className="w-full max-w-md mx-auto" variant="elevated">
      <form onSubmit={handleSubmit}>
        <CardHeader>
          <CardTitle>Sign In</CardTitle>
        </CardHeader>
        <CardContent>
          <InputGroup>
            <Input
              id="email"
              type="email"
              label="Email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              error={errors.email}
              disabled={loading}
              floatingLabel
              icon={
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              }
            />
            <Input
              id="password"
              type="password"
              label="Password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              error={errors.password}
              disabled={loading}
              floatingLabel
              icon={
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              }
            />
          </InputGroup>
        </CardContent>
        <CardFooter className="flex-col gap-4">
          <Button
            type="submit"
            fullWidth
            loading={loading}
            loadingText="Signing in..."
            variant="default"
            size="lg"
          >
            Sign In
          </Button>
          <Button
            type="button"
            variant="link"
            size="sm"
            className="text-muted-foreground"
          >
            Forgot your password?
          </Button>
        </CardFooter>
      </form>
    </Card>
  )
}

// Multi-Step Form Pattern
interface Step {
  id: string
  title: string
  description?: string
}

export const MultiStepFormPattern = () => {
  const steps: Step[] = [
    { id: 'account', title: 'Account', description: 'Create your account' },
    { id: 'profile', title: 'Profile', description: 'Set up your profile' },
    { id: 'preferences', title: 'Preferences', description: 'Choose your preferences' },
    { id: 'review', title: 'Review', description: 'Review and confirm' },
  ]
  
  const [currentStep, setCurrentStep] = React.useState(0)
  const [formData, setFormData] = React.useState({
    email: '',
    password: '',
    name: '',
    bio: '',
    notifications: true,
    theme: 'system',
  })
  
  const isLastStep = currentStep === steps.length - 1
  const isFirstStep = currentStep === 0
  
  const handleNext = () => {
    if (!isLastStep) {
      setCurrentStep(currentStep + 1)
    }
  }
  
  const handlePrevious = () => {
    if (!isFirstStep) {
      setCurrentStep(currentStep - 1)
    }
  }
  
  const handleSubmit = () => {
    console.log('Form submitted:', formData)
  }
  
  return (
    <Card className="w-full max-w-2xl mx-auto" variant="elevated">
      <CardHeader>
        <div className="space-y-4">
          {/* Progress Bar */}
          <div className="relative">
            <div className="flex justify-between mb-2">
              {steps.map((step, index) => (
                <div
                  key={step.id}
                  className={cn(
                    'flex items-center',
                    index !== steps.length - 1 && 'flex-1'
                  )}
                >
                  <motion.div
                    className={cn(
                      'w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium',
                      index <= currentStep
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground'
                    )}
                    initial={false}
                    animate={{
                      scale: index === currentStep ? 1.1 : 1,
                      backgroundColor: index <= currentStep ? 'var(--color-primary)' : 'var(--color-muted)',
                    }}
                    transition={animations.transitions.spring.smooth}
                  >
                    {index + 1}
                  </motion.div>
                  {index !== steps.length - 1 && (
                    <div className="flex-1 mx-2">
                      <div className="h-1 bg-muted rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-primary"
                          initial={false}
                          animate={{
                            scaleX: index < currentStep ? 1 : 0,
                          }}
                          transition={animations.transitions.spring.smooth}
                          style={{ originX: 0 }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Step Info */}
          <div className="text-center">
            <motion.h3
              key={currentStep}
              className="text-xl font-semibold"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={animations.transitions.spring.smooth}
            >
              {steps[currentStep].title}
            </motion.h3>
            {steps[currentStep].description && (
              <motion.p
                key={`${currentStep}-desc`}
                className="text-sm text-muted-foreground mt-1"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
              >
                {steps[currentStep].description}
              </motion.p>
            )}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={animations.transitions.spring.smooth}
        >
          {currentStep === 0 && (
            <InputGroup>
              <Input
                label="Email"
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                floatingLabel
              />
              <Input
                label="Password"
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                floatingLabel
              />
            </InputGroup>
          )}
          
          {currentStep === 1 && (
            <InputGroup>
              <Input
                label="Full Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                floatingLabel
              />
              <div>
                <label className="block text-sm font-medium mb-1.5">Bio</label>
                <textarea
                  className="w-full min-h-[100px] px-4 py-2 border border-input rounded-lg bg-background focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                  value={formData.bio}
                  onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
                  placeholder="Tell us about yourself..."
                />
              </div>
            </InputGroup>
          )}
          
          {currentStep === 2 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 border border-border rounded-lg">
                <div>
                  <p className="font-medium">Email Notifications</p>
                  <p className="text-sm text-muted-foreground">Receive updates via email</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    className="sr-only peer"
                    checked={formData.notifications}
                    onChange={(e) => setFormData({ ...formData, notifications: e.target.checked })}
                  />
                  <div className="w-11 h-6 bg-muted peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
            </div>
          )}
          
          {currentStep === 3 && (
            <div className="space-y-4">
              <h4 className="font-medium">Review Your Information</h4>
              <div className="space-y-2 p-4 bg-muted/50 rounded-lg">
                <p><span className="font-medium">Email:</span> {formData.email || 'Not provided'}</p>
                <p><span className="font-medium">Name:</span> {formData.name || 'Not provided'}</p>
                <p><span className="font-medium">Bio:</span> {formData.bio || 'Not provided'}</p>
                <p><span className="font-medium">Notifications:</span> {formData.notifications ? 'Enabled' : 'Disabled'}</p>
              </div>
            </div>
          )}
        </motion.div>
      </CardContent>
      
      <CardFooter className="justify-between">
        <Button
          onClick={handlePrevious}
          variant="outline"
          disabled={isFirstStep}
        >
          Previous
        </Button>
        <Button
          onClick={isLastStep ? handleSubmit : handleNext}
          variant={isLastStep ? 'default' : 'secondary'}
        >
          {isLastStep ? 'Submit' : 'Next'}
        </Button>
      </CardFooter>
    </Card>
  )
}
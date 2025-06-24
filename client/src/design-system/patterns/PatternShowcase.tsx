/**
 * Pattern Showcase
 * Interactive showcase of design system patterns
 */

import * as React from 'react'
import { motion } from 'framer-motion'
import {
  Button,
  ButtonGroup,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  AnimatedCard,
  CardStack,
  Input,
  SearchInput,
  Modal,
  ModalHeader,
  ModalTitle,
  ModalDescription,
  ModalBody,
  ModalFooter,
  ConfirmationModal,
  Toast,
  useToast,
  toast,
} from '../components'
import { ThemeToggle } from '../ThemeProvider'
import { animations } from '../animations'
import { cn } from '../../utils/cn'

export const PatternShowcase = () => {
  const [activeSection, setActiveSection] = React.useState('buttons')
  const { toast: showToast } = useToast()
  const [modalOpen, setModalOpen] = React.useState(false)
  const [confirmModalOpen, setConfirmModalOpen] = React.useState(false)

  const sections = [
    { id: 'buttons', label: 'Buttons' },
    { id: 'cards', label: 'Cards' },
    { id: 'forms', label: 'Forms' },
    { id: 'modals', label: 'Modals' },
    { id: 'notifications', label: 'Notifications' },
    { id: 'animations', label: 'Animations' },
  ]

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <h1 className="text-2xl font-bold">Design System Patterns</h1>
          <ThemeToggle />
        </div>
      </header>

      {/* Navigation */}
      <nav className="container mt-6">
        <div className="flex gap-2 p-1 bg-muted rounded-lg w-fit">
          {sections.map((section) => (
            <motion.button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={cn(
                'px-4 py-2 rounded-md text-sm font-medium transition-colors',
                activeSection === section.id
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {section.label}
            </motion.button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="container mt-8 pb-16">
        <motion.div
          key={activeSection}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={animations.transitions.spring.smooth}
        >
          {/* Buttons Section */}
          {activeSection === 'buttons' && (
            <div className="space-y-8">
              <Card>
                <CardHeader>
                  <CardTitle>Button Variants</CardTitle>
                  <CardDescription>Different button styles for various use cases</CardDescription>
                </CardHeader>
                <CardContent>
                  <ButtonGroup spacing="md">
                    <Button variant="default">Default</Button>
                    <Button variant="secondary">Secondary</Button>
                    <Button variant="destructive">Destructive</Button>
                    <Button variant="outline">Outline</Button>
                    <Button variant="ghost">Ghost</Button>
                    <Button variant="link">Link</Button>
                    <Button variant="gradient">Gradient</Button>
                    <Button variant="glow">Glow</Button>
                  </ButtonGroup>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Button Sizes</CardTitle>
                  <CardDescription>Multiple size options for different contexts</CardDescription>
                </CardHeader>
                <CardContent>
                  <ButtonGroup spacing="md">
                    <Button size="xs">Extra Small</Button>
                    <Button size="sm">Small</Button>
                    <Button size="default">Default</Button>
                    <Button size="lg">Large</Button>
                    <Button size="xl">Extra Large</Button>
                  </ButtonGroup>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Button States</CardTitle>
                  <CardDescription>Interactive states and loading indicators</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <ButtonGroup spacing="md">
                    <Button loading loadingText="Processing...">Loading</Button>
                    <Button disabled>Disabled</Button>
                    <Button pulse>Pulse Effect</Button>
                    <Button
                      icon={
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                      }
                    >
                      With Icon
                    </Button>
                  </ButtonGroup>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Cards Section */}
          {activeSection === 'cards' && (
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-semibold mb-4">Card Variants</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <Card variant="default">
                    <CardHeader>
                      <CardTitle>Default Card</CardTitle>
                      <CardDescription>Standard card with border</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        This is a default card variant with subtle styling.
                      </p>
                    </CardContent>
                  </Card>

                  <Card variant="elevated">
                    <CardHeader>
                      <CardTitle>Elevated Card</CardTitle>
                      <CardDescription>Card with shadow elevation</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        Elevated cards appear to float above the surface.
                      </p>
                    </CardContent>
                  </Card>

                  <Card variant="outlined">
                    <CardHeader>
                      <CardTitle>Outlined Card</CardTitle>
                      <CardDescription>Card with prominent border</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        Outlined cards have a stronger border presence.
                      </p>
                    </CardContent>
                  </Card>

                  <Card variant="ghost">
                    <CardHeader>
                      <CardTitle>Ghost Card</CardTitle>
                      <CardDescription>Minimal card style</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        Ghost cards have minimal visual presence.
                      </p>
                    </CardContent>
                  </Card>

                  <Card variant="gradient">
                    <CardHeader>
                      <CardTitle>Gradient Card</CardTitle>
                      <CardDescription>Card with gradient background</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        Gradient cards add visual interest.
                      </p>
                    </CardContent>
                  </Card>

                  <Card variant="glass">
                    <CardHeader>
                      <CardTitle>Glass Card</CardTitle>
                      <CardDescription>Glassmorphism effect</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        Glass cards have a translucent appearance.
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div>
                <h2 className="text-2xl font-semibold mb-4">Animated Cards</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <AnimatedCard animationType="lift">
                    <CardHeader>
                      <CardTitle>Lift Animation</CardTitle>
                      <CardDescription>Hover to see lift effect</CardDescription>
                    </CardHeader>
                  </AnimatedCard>

                  <AnimatedCard animationType="glow">
                    <CardHeader>
                      <CardTitle>Glow Animation</CardTitle>
                      <CardDescription>Hover to see glow effect</CardDescription>
                    </CardHeader>
                  </AnimatedCard>

                  <AnimatedCard animationType="tilt">
                    <CardHeader>
                      <CardTitle>Tilt Animation</CardTitle>
                      <CardDescription>Hover to see tilt effect</CardDescription>
                    </CardHeader>
                  </AnimatedCard>
                </div>
              </div>
            </div>
          )}

          {/* Forms Section */}
          {activeSection === 'forms' && (
            <div className="space-y-8">
              <Card>
                <CardHeader>
                  <CardTitle>Input Variants</CardTitle>
                  <CardDescription>Different input styles and states</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    label="Default Input"
                    placeholder="Enter text..."
                    helper="This is a helper text"
                  />
                  <Input
                    variant="filled"
                    label="Filled Input"
                    placeholder="Enter text..."
                  />
                  <Input
                    variant="flushed"
                    label="Flushed Input"
                    placeholder="Enter text..."
                  />
                  <Input
                    variant="outlined"
                    label="Outlined Input"
                    placeholder="Enter text..."
                  />
                  <Input
                    label="Input with Error"
                    placeholder="Enter text..."
                    error="This field is required"
                  />
                  <Input
                    label="Input with Success"
                    placeholder="Enter text..."
                    success="Looking good!"
                  />
                  <SearchInput
                    placeholder="Search..."
                    onSearch={(value) => console.log('Search:', value)}
                  />
                </CardContent>
              </Card>
            </div>
          )}

          {/* Modals Section */}
          {activeSection === 'modals' && (
            <div className="space-y-8">
              <Card>
                <CardHeader>
                  <CardTitle>Modal Examples</CardTitle>
                  <CardDescription>Different modal types and positions</CardDescription>
                </CardHeader>
                <CardContent>
                  <ButtonGroup>
                    <Button onClick={() => setModalOpen(true)}>Open Modal</Button>
                    <Button
                      variant="destructive"
                      onClick={() => setConfirmModalOpen(true)}
                    >
                      Confirmation Modal
                    </Button>
                  </ButtonGroup>

                  <Modal
                    open={modalOpen}
                    onOpenChange={setModalOpen}
                    size="lg"
                  >
                    <ModalHeader>
                      <ModalTitle>Example Modal</ModalTitle>
                      <ModalDescription>
                        This is an example modal with header, body, and footer sections.
                      </ModalDescription>
                    </ModalHeader>
                    <ModalBody>
                      <p className="text-sm text-muted-foreground">
                        Modal content goes here. You can add any content including forms,
                        images, or other components.
                      </p>
                    </ModalBody>
                    <ModalFooter>
                      <Button variant="outline" onClick={() => setModalOpen(false)}>
                        Cancel
                      </Button>
                      <Button onClick={() => setModalOpen(false)}>
                        Save Changes
                      </Button>
                    </ModalFooter>
                  </Modal>

                  <ConfirmationModal
                    open={confirmModalOpen}
                    onOpenChange={setConfirmModalOpen}
                    onConfirm={() => console.log('Confirmed!')}
                    title="Are you sure?"
                    description="This action cannot be undone. Please confirm to proceed."
                    type="danger"
                    confirmText="Delete"
                  />
                </CardContent>
              </Card>
            </div>
          )}

          {/* Notifications Section */}
          {activeSection === 'notifications' && (
            <div className="space-y-8">
              <Card>
                <CardHeader>
                  <CardTitle>Toast Notifications</CardTitle>
                  <CardDescription>Different toast variants and positions</CardDescription>
                </CardHeader>
                <CardContent>
                  <ButtonGroup>
                    <Button
                      variant="outline"
                      onClick={() => showToast({
                        title: 'Default Toast',
                        description: 'This is a default toast notification',
                      })}
                    >
                      Default Toast
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => showToast(toast.success('Success!', 'Operation completed successfully'))}
                    >
                      Success Toast
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => showToast(toast.error('Error!', 'Something went wrong'))}
                    >
                      Error Toast
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => showToast(toast.warning('Warning!', 'Please review your input'))}
                    >
                      Warning Toast
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => showToast(toast.info('Info', 'Here is some information'))}
                    >
                      Info Toast
                    </Button>
                  </ButtonGroup>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Animations Section */}
          {activeSection === 'animations' && (
            <div className="space-y-8">
              <Card>
                <CardHeader>
                  <CardTitle>Animation Presets</CardTitle>
                  <CardDescription>Built-in animation utilities</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(animations.variants).map(([name, variant]) => (
                      <motion.div
                        key={name}
                        className="p-4 bg-muted rounded-lg text-center"
                        variants={variant}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                      >
                        <p className="text-sm font-medium">{name}</p>
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </motion.div>
      </main>
    </div>
  )
}
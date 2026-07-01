import React from 'react';
import { screen } from '@testing-library/react';
import { render } from './test-utils';
import { Base } from './Base';

test('renders login screen', () => {
	render(<Base />);
	const signInElements = screen.getAllByText(/sign in/i);
	expect(signInElements.length).toBeGreaterThan(0);
});

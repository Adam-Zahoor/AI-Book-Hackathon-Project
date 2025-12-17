import React from 'react';
import Chatbot from '@site/src/components/Chatbot';

/**
 * Component that wraps the children components with the Chatbot
 */
export default function LayoutWrapper(props) {
  const { children } = props;
  return (
    <>
      {children}
      <Chatbot />
    </>
  );
}
import { useState } from "react";
import { Button } from "@humansignal/ui";
import { ErrorWrapper } from "../../../components/Error/Error";
import { InlineError } from "../../../components/Error/InlineError";
import { Form, Input, Select, TextArea, Toggle } from "../../../components/Form";
import "./MachineLearningSettings.prefix.css";

const CustomBackendForm = ({ action, backend, project, onSubmit }) => {
  const [selectedAuthMethod, setAuthMethod] = useState("NONE");
  const [, setMLError] = useState();
  const [isLocal, setIsLocal] = useState(Boolean(backend?.is_local ?? false));

  // Toggle의 onChange가 boolean이 아니라 이벤트 객체(e.target.checked)나
  // 다른 값을 넘겨주는 경우까지 방어적으로 처리하여, 항상 순수 boolean만
  // state와 폼에 저장되도록 정규화한다.
  const handleIsLocalChange = (val) => {
    let boolValue;
    if (typeof val === "boolean") {
      boolValue = val;
    } else if (val && typeof val === "object" && "target" in val) {
      boolValue = Boolean(val.target.checked);
    } else {
      boolValue = Boolean(val);
    }
    setIsLocal(boolValue);
  };

  return (
    <Form
      action={action}
      formData={{ ...(backend ?? {}) }}
      params={{ pk: backend?.id }}
      onSubmit={async (response) => {
        if (!response.error_message) {
          onSubmit(response);
        }
      }}
    >
      <Input type="hidden" name="project" value={project.id} />

      <Form.Row columnCount={1}>
        <Input name="title" label="Name" placeholder="Enter a name" required />
      </Form.Row>

      <Form.Row columnCount={1}>
        <Toggle
          name="is_local"
          label="Local Model"
          description="If enabled this is a Local Model."
          value={isLocal}
          onChange={handleIsLocalChange}
        />
      </Form.Row>

      <Form.Row columnCount={1}>
        <Input
          name="url"
          label={isLocal ? "Local Path" : "Backend URL"}
          placeholder={isLocal ? "/path/to/model" : "http://localhost:9090"}
          required
        />
      </Form.Row>

      <Form.Row columnCount={2}>
        <Select
          name="auth_method"
          label="Select authentication method"
          options={[
            { label: "No Authentication", value: "NONE" },
            { label: "Basic Authentication", value: "BASIC_AUTH" },
          ]}
          value={selectedAuthMethod}
          onChange={setAuthMethod}
        />
      </Form.Row>

      {(backend?.auth_method === "BASIC_AUTH" || selectedAuthMethod === "BASIC_AUTH") && (
        <Form.Row columnCount={2}>
          <Input name="basic_auth_user" label="Basic auth user" />
          {backend?.basic_auth_pass_is_set ? (
            <Input name="basic_auth_pass" label="Basic auth pass" type="password" placeholder="********" />
          ) : (
            <Input name="basic_auth_pass" label="Basic auth pass" type="password" />
          )}
        </Form.Row>
      )}

      <Form.Row columnCount={1}>
        <TextArea
          name="extra_params"
          label="Any extra params to pass during model connection"
          style={{ minHeight: 120 }}
        />
      </Form.Row>

      <Form.Row columnCount={1}>
        <Toggle
          name="is_interactive"
          label="Interactive preannotations"
          description="If enabled some labeling tools will send requests to the ML Backend interactively during the annotation process."
        />
      </Form.Row>

      <Form.Actions>
        <Button type="submit" look="primary" onClick={() => setMLError(null)} aria-label="Save machine learning form">
          Validate and Save
        </Button>
      </Form.Actions>

      <Form.ResponseParser>
        {(response) => (
          <>
            {response.error_message && (
              <ErrorWrapper
                error={{
                  response: {
                    detail: `Failed to ${backend ? "save" : "add new"} ML backend.`,
                    exc_info: response.error_message,
                  },
                }}
              />
            )}
          </>
        )}
      </Form.ResponseParser>

      <InlineError />
    </Form>
  );
};

export { CustomBackendForm };

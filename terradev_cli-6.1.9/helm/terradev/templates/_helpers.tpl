{{/*
Expand the name of the chart.
*/}}
{{- define "terradev.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "terradev.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "terradev.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "terradev.labels" -}}
helm.sh/chart: {{ include "terradev.chart" . }}
{{ include "terradev.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "terradev.selectorLabels" -}}
app.kubernetes.io/name: {{ include "terradev.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "terradev.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "terradev.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Common annotations
*/}}
{{- define "terradev.annotations" -}}
{{- with .Values.extraAnnotations }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "terradev.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if .Values.image.digest }}
{{- printf "%s/%s@%s" $registry $repository .Values.image.digest -}}
{{- else }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "terradev.imagePullSecrets" -}}
{{- include "common.images.pullSecrets" (dict "images" (list .Values.image) "global" .Values.global) -}}
{{- end }}

{{/*
Validate values
*/}}
{{- define "terradev.validateValues" -}}
{{- $messages := list -}}
{{- $messages := append $messages (include "terradev.validateValues.database" .) -}}
{{- $messages := append $messages (include "terradev.validateValues.redis" .) -}}
{{- $messages := without $messages "" -}}
{{- $message := join "\n" $messages -}}
{{- if $message -}}
{{- printf "\nVALUES VALIDATION:\n%s" $message -}}
{{- end -}}
{{- end }}

{{/*
Validate database values
*/}}
{{- define "terradev.validateValues.database" -}}
{{- if and (not .Values.postgresql.enabled) (not .Values.externalDatabase.host) -}}
terradev: Database configuration is required. Either enable PostgreSQL or configure an external database.
{{- end -}}
{{- end }}

{{/*
Validate redis values
*/}}
{{- define "terradev.validateValues.redis" -}}
{{- if and (not .Values.redis.enabled) (not .Values.externalRedis.host) -}}
terradev: Redis configuration is required. Either enable Redis or configure an external Redis instance.
{{- end -}}
{{- end }}
